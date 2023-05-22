from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multitask = torch.load("./results/models/MultiTaskHead.pth")
head = MultiTaskHead()

head.label = multitask.label
head.gender = multitask.gender
head.gender_conv = multitask.gender_conv
head.sleeve = multitask.sleeve
head.sleeve_conv = multitask.sleeve_conv
head.pov = multitask.pov
head.pov_conv = multitask.pov_conv
head.avg_pool =  multitask.avg_pool

torch.save(head, "./results/models/multi-label-head.pth")


