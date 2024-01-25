from toy_modules import *

label_possibilities = [F.one_hot(torch.tensor(k), 4).float().cuda() for k in range(4)]
l1_loss = nn.L1Loss().cuda()

toy_lightning = ToyLightning.load_from_checkpoint('C:/Users/adria/Desktop/Master topic/Repository/main_model/tests/runs/test_toy_example/l5hhgi6q/checkpoints/epoch=299-step=42300.ckpt')

datapoints = torch.zeros((40000, 2)).cuda()
for counter1, i in enumerate([x/10-10 for x in range(200)]):
    for counter2, j in enumerate([x/10-10 for x in range(200)]):
        datapoints[counter1*200+counter2] = torch.tensor((i, j)).cuda()

reconstruction = torch.zeros((4, 40000, 2)).cuda()
for i, label_possibility in enumerate(label_possibilities):
    reconstruction[i], _ = toy_lightning(datapoints, condition_vector=label_possibility)

print(reconstruction.size())

closed_datapoints = torch.zeros((40000, 2))
open_datapoints = torch.zeros((40000, 2))

for i in range(40000):
    loss_list = []
    for j in range(4):
        loss_list.append(l1_loss(reconstruction[j, i], datapoints[i]))
    if min(loss_list) < 2.5:
        closed_datapoints[i] = datapoints[i]
    else:
        open_datapoints[i] = datapoints[i]

plt.scatter(x=np.array(closed_datapoints[:, 0]), y=np.array(closed_datapoints[:, 1]), label='closed_points')
plt.scatter(x=np.array(open_datapoints[:, 0]), y=np.array(open_datapoints[:, 1]), label='open_points')
plt.legend()
plt.show()
