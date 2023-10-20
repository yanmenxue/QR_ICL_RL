import torch

# train_data = []
# with open('../QR_dataset/canard/train.txt', 'r', encoding='utf-8') as f_train:
#   for line in f_train:
#     line = line.strip().split('\t\t')
#     data = {}
#     context =''
#     for j in range(len(line)-2):
#       if j != len(line) - 3:
#         context += line[j] + '; '
#       else:
#         context += line[j]
#     cur = line[-2]
#     rewrite = line[-1]
#     data['context'] = context
#     data['cur'] = cur
#     data['rewrite'] = rewrite
#     train_data.append(data)
#
# torch.save(train_data, 'canard_train')
# print(len(train_data))
# print(train_data[0])
#
# dev_data = []
# with open('../QR_dataset/canard/dev.txt', 'r', encoding='utf-8') as f_dev:
#   for line in f_dev:
#     line = line.strip().split('\t\t')
#     data = {}
#     context =''
#     for j in range(len(line)-2):
#       if j != len(line) - 3:
#         context += line[j] + '; '
#       else:
#         context += line[j]
#     cur = line[-2]
#     rewrite = line[-1]
#     data['context'] = context
#     data['cur'] = cur
#     data['rewrite'] = rewrite
#     dev_data.append(data)
# torch.save(dev_data, 'canard_dev')
# print(dev_data[0])
#
# test_data = []
# with open('../QR_dataset/canard/test.txt', 'r', encoding='utf-8') as f_test:
#   for line in f_test:
#     line = line.strip().split('\t\t')
#     data = {}
#     context = ''
#     for j in range(len(line)-2):
#       if j != len(line) - 3:
#         context += line[j] + '; '
#       else:
#         context += line[j]
#     cur = line[-2]
#     rewrite = line[-1]
#     data['context'] = context
#     data['cur'] = cur
#     data['rewrite'] = rewrite
#     test_data.append(data)
# torch.save(test_data, 'canard_test')
# print(test_data[0])


# train_data = []
# with open('../QR_dataset/multi/train.txt', 'r', encoding='utf-8') as f_train:
#   for line in f_train:
#     line = line.strip().split('\t\t')
#     data = {}
#     context =''
#     for j in range(len(line)-2):
#       if j != len(line) - 3:
#         context += line[j] + '; '
#       else:
#         context += line[j]
#     cur = line[-2]
#     rewrite = line[-1]
#     data['context'] = context
#     data['cur'] = cur
#     data['rewrite'] = rewrite
#     train_data.append(data)
#
# torch.save(train_data, 'multi_train')
# print(len(train_data))
# print(train_data[0])
#
# dev_data = []
# with open('../QR_dataset/multi/valid.txt', 'r', encoding='utf-8') as f_dev:
#   for line in f_dev:
#     line = line.strip().split('\t\t')
#     data = {}
#     context =''
#     for j in range(len(line)-2):
#       if j != len(line) - 3:
#         context += line[j] + '; '
#       else:
#         context += line[j]
#     cur = line[-2]
#     rewrite = line[-1]
#     data['context'] = context
#     data['cur'] = cur
#     data['rewrite'] = rewrite
#     dev_data.append(data)
# torch.save(dev_data, 'multi_dev')
# print(dev_data[0])
#
# test_data = []
# with open('../QR_dataset/multi/test.txt', 'r', encoding='utf-8') as f_test:
#   for line in f_test:
#     line = line.strip().split('\t\t')
#     data = {}
#     context = ''
#     for j in range(len(line)-2):
#       if j != len(line) - 3:
#         context += line[j] + '; '
#       else:
#         context += line[j]
#     cur = line[-2]
#     rewrite = line[-1]
#     data['context'] = context
#     data['cur'] = cur
#     data['rewrite'] = rewrite
#     test_data.append(data)
# torch.save(test_data, 'multi_test')
# print(test_data[0])


# train_data = []
# with open('../QR_dataset/task/train.txt', 'r', encoding='utf-8') as f_train:
#   for line in f_train:
#     line = line.strip().split('\t\t')
#     data = {}
#     context =''
#     for j in range(len(line)-2):
#       if j != len(line) - 3:
#         context += line[j] + '; '
#       else:
#         context += line[j]
#     cur = line[-2]
#     rewrite = line[-1]
#     data['context'] = context
#     data['cur'] = cur
#     data['rewrite'] = rewrite
#     train_data.append(data)
#
# torch.save(train_data, 'task_train')
# print(len(train_data))
# print(train_data[0])
#
# dev_data = []
# with open('../QR_dataset/task/dev.txt', 'r', encoding='utf-8') as f_dev:
#   for line in f_dev:
#     line = line.strip().split('\t\t')
#     data = {}
#     context =''
#     for j in range(len(line)-2):
#       if j != len(line) - 3:
#         context += line[j] + '; '
#       else:
#         context += line[j]
#     cur = line[-2]
#     rewrite = line[-1]
#     data['context'] = context
#     data['cur'] = cur
#     data['rewrite'] = rewrite
#     dev_data.append(data)
# torch.save(dev_data, 'task_dev')
# print(dev_data[0])


train_data = []
with open('../QR_dataset/rewrite/train.txt', 'r', encoding='utf-8') as f_train:
  for line in f_train:
    line = line.strip().split('\t\t')
    data = {}
    context =''
    for j in range(len(line)-2):
      if j != len(line) - 3:
        context += line[j] + '; '
      else:
        context += line[j]
    cur = line[-2]
    rewrite = line[-1]
    data['context'] = context
    data['cur'] = cur
    data['rewrite'] = rewrite
    train_data.append(data)

torch.save(train_data, 'rewrite_train')
print(len(train_data))
print(train_data[0])

dev_data = []
with open('../QR_dataset/rewrite/dev.txt', 'r', encoding='utf-8') as f_dev:
  for line in f_dev:
    line = line.strip().split('\t\t')
    data = {}
    context =''
    for j in range(len(line)-2):
      if j != len(line) - 3:
        context += line[j] + '; '
      else:
        context += line[j]
    cur = line[-2]
    rewrite = line[-1]
    data['context'] = context
    data['cur'] = cur
    data['rewrite'] = rewrite
    dev_data.append(data)
torch.save(dev_data, 'rewrite_dev')
print(dev_data[0])