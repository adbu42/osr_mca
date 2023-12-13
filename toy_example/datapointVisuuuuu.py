from toy_modules import *


toy_dataset_train, toy_dataset_valid = random_split(ToyDataset(), [18000, 2000])


datapoint_visu_0 = torch.zeros((2000, 2))
datapoint_visu_1 = torch.zeros((2000, 2))
datapoint_visu_2 = torch.zeros((2000, 2))
datapoint_visu_3 = torch.zeros((2000, 2))
i0 = 0
i1 = 0
i2 = 0
i3 = 0
for datapoint, classification, _, _ in iter(toy_dataset_valid):
    if classification == 0:
        datapoint_visu_0[i0] = datapoint
        i0 += 1
    if classification == 1:
        datapoint_visu_1[i1] = datapoint
        i1 += 1
    if classification == 2:
        datapoint_visu_2[i2] = datapoint
        i2 += 1
    if classification == 3:
        datapoint_visu_3[i3] = datapoint
        i3 += 1

plt.scatter(x=np.array(datapoint_visu_0[:, 0]), y=np.array(datapoint_visu_0[:, 1]))
plt.scatter(x=np.array(datapoint_visu_1[:, 0]), y=np.array(datapoint_visu_1[:, 1]))
plt.scatter(x=np.array(datapoint_visu_2[:, 0]), y=np.array(datapoint_visu_2[:, 1]))
plt.scatter(x=np.array(datapoint_visu_3[:, 0]), y=np.array(datapoint_visu_3[:, 1]))
plt.show()

toy_lightning = ToyLightning.load_from_checkpoint('tests/runs/test_toy_example/5029y1lj/checkpoints/epoch=49-step=7050.ckpt')

datapoint_visu_0 = torch.zeros((2000, 2))
datapoint_visu_1 = torch.zeros((2000, 2))
datapoint_visu_2 = torch.zeros((2000, 2))
datapoint_visu_3 = torch.zeros((2000, 2))
i0 = 0
i1 = 0
i2 = 0
i3 = 0
for datapoint, label, _, _ in iter(toy_dataset_valid):
    reconstruction, classifications = toy_lightning(datapoint.cuda(), condition_vector=F.one_hot(torch.tensor(label), 4).float().cuda())
    classifications_max = torch.argmax(classifications)
    if classifications_max == 0:
        datapoint_visu_0[i0] = datapoint
        i0 += 1
    if classifications_max == 1:
        datapoint_visu_1[i1] = datapoint
        i1 += 1
    if classifications_max == 2:
        datapoint_visu_2[i2] = datapoint
        i2 += 1
    if classifications_max == 3:
        datapoint_visu_3[i3] = datapoint
        i3 += 1
plt.scatter(x=np.array(datapoint_visu_0[:, 0]), y=np.array(datapoint_visu_0[:, 1]))
plt.scatter(x=np.array(datapoint_visu_1[:, 0]), y=np.array(datapoint_visu_1[:, 1]))
plt.scatter(x=np.array(datapoint_visu_2[:, 0]), y=np.array(datapoint_visu_2[:, 1]))
plt.scatter(x=np.array(datapoint_visu_3[:, 0]), y=np.array(datapoint_visu_3[:, 1]))
plt.show()
