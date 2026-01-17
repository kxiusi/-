import torch
batch = [[1],[2],[3],[4],[5]]
batch=torch.LongTensor(batch)
print(batch)
#torch.nn.Embedding(num_embeddings: int, embedding_dim: int)
#num_embeddings表示嵌入的词个数，如果输入的的是数组，那么num_embeddings至少要比数组中最大的元素要大
#否则，会出现IndexError: index out of range in self
embed=torch.nn.Embedding(6,6)
embed_batch = embed(batch)
print(embed_batch)
#embed.weight[1:]就是embedding层的输出结果
print(embed.weight[1:])