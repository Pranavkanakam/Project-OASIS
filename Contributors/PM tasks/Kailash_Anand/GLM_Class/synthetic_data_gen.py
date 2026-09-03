import numpy as np
import torch
import csv

torch.manual_seed(0)

N = 100
P = 2

noise_mean = 0.0
noise_std = 0.05
        

b = torch.tensor([0.5, 3, 0]).view(-1, 1) # goes from b_0, b_1, b_2...b_P

if (b.shape[0] != P+1):
    print("Size mis-match")
    quit()

X = torch.rand(size=(N,P+1))
X[:,0] = torch.ones(N)

e = torch.normal(noise_mean, noise_std, (N,)).view(-1, 1)
y = X @ b



data = torch.cat((X[:,1:], y), dim=1).numpy()
print(data[:5])

with open('/Users/kailashanand/Documents/Developer/MC_OASIS/Task_Week_7/amazing_data_0.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

# import numpy as np
# import torch
# import csv


# b_2 = 0
# b_1 = 3
# b_0 = 0.5

# x_1 = torch.rand(100).view(-1,1) 
# x_2 = torch.rand(100).view(-1,1) 

# e = torch.normal(0,0.4,(100,)).view(-1,1)

# y = b_0 + b_1*x_1 + b_2*x_2 + e


# data = torch.cat((x_1, x_2, y), dim=1).numpy()
# print(data[:5])


# with open('/Users/kailashanand/Documents/Developer/MC_OASIS/Task_Week6/amazing_data_0.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(data)
