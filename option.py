import argparse

parser = argparse.ArgumentParser(description='MGN')

parser.add_argument('--nThread', type=int, default=2, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs')

parser.add_argument("--datadir", type=str, default="Market-1501-v15.09.15", help='dataset directory')
parser.add_argument('--data_train', type=str, default='Market1501', help='train dataset name')
parser.add_argument('--data_test', type=str, default='Market1501', help='test dataset name')

parser.add_argument('--reset', action='store_true', help='reset the training')
parser.add_argument("--epochs", type=int, default=80, help='number of epochs to train')
parser.add_argument('--test_every', type=int, default=20, help='do test per every N epochs')
parser.add_argument("--batchid", type=int, default=16, help='the batch for id')
parser.add_argument("--batchimage", type=int, default=4, help='the batch of per id')
parser.add_argument("--batchtest", type=int, default=32, help='input batch size for test')
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')

parser.add_argument('--model', default='MGN', help='model name')
parser.add_argument('--loss', type=str, default='1*CrossEntropy+1*Triplet', help='loss function configuration')

parser.add_argument('--act', type=str, default='relu', help='activation function')
parser.add_argument('--pool', type=str, default='avg', help='pool function')
parser.add_argument('--feats', type=int, default=256, help='number of feature maps')
parser.add_argument('--height', type=int, default=384, help='height of the input image')
parser.add_argument('--width', type=int, default=128, help='width of the input image')
parser.add_argument('--num_classes', type=int, default=751, help='')


parser.add_argument("--lr", type=float, default=2e-4, help='learning rate')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD','ADAM','NADAM','RMSprop'), help='optimizer to use (SGD | ADAM | NADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--dampening', type=float, default=0, help='SGD dampening')
parser.add_argument('--nesterov', action='store_true', help='SGD nesterov')
parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2')
parser.add_argument('--amsgrad', action='store_true', help='ADAM amsgrad')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor for step decay')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
parser.add_argument('--lr_decay', type=int, default=60, help='learning rate decay per N epochs')

parser.add_argument("--margin", type=float, default=1.2, help='')
parser.add_argument("--re_rank", action='store_true', help='')
parser.add_argument("--random_erasing", action='store_true', help='')
parser.add_argument("--probability", type=float, default=0.5, help='')

parser.add_argument("--savedir", type=str, default='saved_models', help='directory name to save')
parser.add_argument("--outdir", type=str, default='out', help='')
parser.add_argument("--resume", type=int, default=0, help='resume from specific checkpoint')
parser.add_argument('--save', type=str, default='test', help='file name to save')
parser.add_argument('--load', type=str, default='', help='file name to load')
parser.add_argument('--save_models', action='store_true', help='save all intermediate models')
parser.add_argument('--pre_train', type=str, default='', help='pre-trained model directory')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

