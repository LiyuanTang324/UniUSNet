import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config

from datasets.dataset import CenterCropGenerator
from datasets.dataset import USdatasetCls, USdatasetSeg

from utils import omni_seg_test
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

from networks.omni_vision_transformer import OmniVisionTransformer as ViT_omni

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../../data', help='root dir for data')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_epochs', type=int, default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
parser.add_argument('--is_saveout', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default="configs/swin_tiny_patch4_window8_256_lite.yaml",
                    metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

parser.add_argument('--prompt', action='store_true', help='using prompt')

args = parser.parse_args()
config = get_config(args)


def discover_dataset_names(root_path, task_name, split):
    task_root = os.path.join(root_path, task_name)
    if not os.path.isdir(task_root):
        return []
    dataset_names = []
    for dataset_name in sorted(os.listdir(task_root)):
        dataset_dir = os.path.join(task_root, dataset_name)
        split_file = os.path.join(dataset_dir, f"{split}.txt")
        if os.path.isdir(dataset_dir) and os.path.exists(split_file):
            dataset_names.append(dataset_name)
    return dataset_names


def inference(args, model, test_save_path=None):
    import csv
    import time

    if not os.path.exists("exp_out/result.csv"):
        with open("exp_out/result.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['dataset', 'task', 'metric', 'time'])

    seg_test_set = discover_dataset_names(args.root_path, "segmentation", "test")

    for dataset_name in seg_test_set:
        num_classes = 2
        db_test = USdatasetSeg(
            base_dir=os.path.join(args.root_path, "segmentation", dataset_name),
            split="test",
            list_dir=os.path.join(args.root_path, "segmentation", dataset_name),
            transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
            prompt=args.prompt
        )
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
        logging.info("{} test iterations per epoch".format(len(testloader)))
        model.eval()

        dice_list = 0.0
        iou_list = 0.0
        hd95_list = 0.0
        count_matrix = np.ones((len(db_test), num_classes-1))
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            if args.prompt:
                position_prompt = sampled_batch['position_prompt'].float()
                task_prompt = torch.FloatTensor([[1, 0]]).expand(position_prompt.shape[0], -1)
                type_prompt = sampled_batch['type_prompt'].float()
                nature_prompt = sampled_batch['nature_prompt'].float()
                metric_i = omni_seg_test(image, label, model,
                                         classes=num_classes,
                                         test_save_path=test_save_path,
                                         case=case_name,
                                         prompt=args.prompt,
                                         type_prompt=type_prompt,
                                         nature_prompt=nature_prompt,
                                         position_prompt=position_prompt,
                                         task_prompt=task_prompt
                                         )
            else:
                metric_i = omni_seg_test(image, label, model,
                                         classes=num_classes,
                                         test_save_path=test_save_path,
                                         case=case_name)
            zero_label_flag = False
            for i in range(1, num_classes):
                if not metric_i[i-1][3]:
                    count_matrix[i_batch, i-1] = 0
                    zero_label_flag = True
            dice_vals = [element[0] for element in metric_i]
            iou_vals = [element[1] for element in metric_i]
            hd95_vals = [element[2] for element in metric_i]
            dice_list += np.array(dice_vals)
            iou_list += np.array(iou_vals)
            hd95_list += np.array(hd95_vals)
            logging.info('idx %d case %s dice %f iou %f hd95 %f' %
                         (i_batch, case_name, np.mean(dice_vals), np.mean(iou_vals), np.mean(hd95_vals)))
            logging.info("This case has zero label: %s" % zero_label_flag)

        valid_count = count_matrix.sum(axis=0) + 1e-6
        dice_list = dice_list / valid_count
        iou_list = iou_list / valid_count
        hd95_list = hd95_list / valid_count
        for i in range(1, num_classes):
            logging.info('Mean class %d  DSC %f  IoU %f  HD95 %f' % (i, dice_list[i-1], iou_list[i-1], hd95_list[i-1]))
        mean_dice = np.mean(dice_list, axis=0)
        mean_iou = np.mean(iou_list, axis=0)
        mean_hd95 = np.mean(hd95_list, axis=0)
        logging.info('Testing seg performance: DSC=%f  IoU=%f  HD95=%f' % (mean_dice, mean_iou, mean_hd95))

        task_tag = 'omni_seg_prompt@' if args.prompt else 'omni_seg@'
        with open("exp_out/result.csv", 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([dataset_name, task_tag + args.output_dir, mean_dice,
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())])

    cls_test_set = discover_dataset_names(args.root_path, "classification", "test")

    for dataset_name in cls_test_set:
        num_classes = 2
        db_test = USdatasetCls(
            base_dir=os.path.join(args.root_path, "classification", dataset_name),
            split="test",
            list_dir=os.path.join(args.root_path, "classification", dataset_name),
            transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
            prompt=args.prompt
        )

        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
        logging.info("{} test iterations per epoch".format(len(testloader)))
        model.eval()

        label_list = []
        prediction_prob_list = []
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            if args.prompt:
                position_prompt = sampled_batch['position_prompt'].float()
                task_prompt = torch.FloatTensor([[0, 1]]).expand(position_prompt.shape[0], -1)
                type_prompt = sampled_batch['type_prompt'].float()
                nature_prompt = sampled_batch['nature_prompt'].float()
                with torch.no_grad():
                    output = model((image.cuda(), position_prompt.cuda(), task_prompt.cuda(),
                                   type_prompt.cuda(), nature_prompt.cuda()))[1]
            else:
                with torch.no_grad():
                    output = model(image.cuda())[1]

            output_prob = torch.softmax(output, dim=1).data.cpu().numpy()
            pred_cls = np.argmax(output_prob)
            logging.info('idx %d case %s label: %d predict: %d' % (i_batch, case_name, label, pred_cls))

            label_list.append(label.numpy())
            prediction_prob_list.append(output_prob)

        labels_flat = np.array(label_list).flatten().astype('uint8')
        probs_all = np.concatenate(prediction_prob_list, axis=0)
        preds_flat = np.argmax(probs_all, axis=1)

        try:
            label_onehot = np.eye(num_classes)[labels_flat]
            auc_val = roc_auc_score(label_onehot, probs_all, multi_class='ovo')
        except ValueError:
            auc_val = 0.5

        try:
            macro_f1 = f1_score(labels_flat, preds_flat, average='macro')
        except ValueError:
            macro_f1 = 0.0

        try:
            pos_probs = probs_all[:, 1]
            fpr, tpr, _ = roc_curve(labels_flat, pos_probs)
            specificity_arr = 1.0 - fpr
            valid_90 = np.where(specificity_arr >= 0.90)[0]
            sens_at_spec90 = tpr[valid_90[-1]] if len(valid_90) > 0 else 0.0
            valid_95 = np.where(specificity_arr >= 0.95)[0]
            sens_at_spec95 = tpr[valid_95[-1]] if len(valid_95) > 0 else 0.0
        except (ValueError, IndexError):
            sens_at_spec90 = 0.0
            sens_at_spec95 = 0.0

        logging.info('--- %s Classification Results ---' % dataset_name)
        logging.info('  AUC:            %f' % auc_val)
        logging.info('  Macro F1-Score: %f' % macro_f1)
        logging.info('  Sens@Spec90:    %f' % sens_at_spec90)
        logging.info('  Sens@Spec95:    %f' % sens_at_spec95)

        task_tag = 'omni_cls_prompt@' if args.prompt else 'omni_cls@'
        with open("exp_out/result.csv", 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([dataset_name, task_tag + args.output_dir, auc_val,
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())])


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    net = ViT_omni(
        config,
        prompt=args.prompt,
    ).cuda()
    net.load_from(config)

    snapshot = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(snapshot):
        snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))

    device = torch.device("cuda")
    model = net.to(device=device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    torch.distributed.init_process_group(backend="nccl", init_method='env://', world_size=1, rank=0)
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    import copy
    pretrained_dict = torch.load(snapshot, map_location=device)
    full_dict = copy.deepcopy(pretrained_dict)
    for k, v in pretrained_dict.items():
        if "module." not in k:
            full_dict["module."+k] = v
            del full_dict[k]

    msg = model.load_state_dict(full_dict)

    print("self trained swin unet", msg)
    snapshot_name = snapshot.split('/')[-1]

    logging.basicConfig(filename=args.output_dir+"/"+"test_result.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_saveout:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)
