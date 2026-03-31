import numpy as np
import cebra
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# load cleaned data from part 1
data = np.load('cleaned_data.npz')
pos_A, neg_A = data['pos_A'], data['neg_A']
pos_B, neg_B = data['pos_B'], data['neg_B']


# make sure both participants same length
# avoids mismatch when combining
min_pos = min(len(pos_A), len(pos_B))
min_neg = min(len(neg_A), len(neg_B))


# combine A + B along channels
# now treating both brains together
pos_combined = np.concatenate([pos_A[:min_pos], pos_B[:min_pos]], axis=1)
neg_combined = np.concatenate([neg_A[:min_neg], neg_B[:min_neg]], axis=1)


# normalize each channel
# remove scale differences
pos_combined = (pos_combined - pos_combined.mean(axis=0)) / (pos_combined.std(axis=0) + 1e-8)
neg_combined = (neg_combined - neg_combined.mean(axis=0)) / (neg_combined.std(axis=0) + 1e-8)


# stack +ve + -ve together
X = np.concatenate([pos_combined, neg_combined], axis=0).astype(np.float32)
y = np.concatenate([np.zeros(len(pos_combined)), np.ones(len(neg_combined))]).astype(int)


# shuffle so model doesn’t see ordered data
idx = np.random.RandomState(42).permutation(len(X))
X, y = X[idx], y[idx]


# setup CEBRA model
model = cebra.CEBRA(
    model_architecture='offset10-model',
    batch_size=512,
    learning_rate=3e-4,
    temperature=1.0,
    output_dimension=3,
    max_iterations=1000,
    device='cpu',
    verbose=True
)


# train embedding
# tries to separate pos vs neg
model.fit(X, y)


# get 3D embedding
embedding = model.transform(X)


# simple classifier on embedding
# check if separation is real
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, embedding, y, cv=5)
print(f"KNN Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")


# shuffled labels → sanity check
# if same accuracy → model learned nothing
y_shuffled = y.copy()
np.random.RandomState(99).shuffle(y_shuffled)


# train again on fake labels
model_s = cebra.CEBRA(
    model_architecture='offset10-model',
    batch_size=512,
    learning_rate=3e-4,
    temperature=1.0,
    output_dimension=3,
    max_iterations=1000,
    device='cpu',
    verbose=True
)
model_s.fit(X, y_shuffled)


# embedding with wrong labels
embedding_s = model_s.transform(X)


scores_s = cross_val_score(knn, embedding_s, y_shuffled, cv=5)
print(f"Shuffled KNN Accuracy: {scores_s.mean():.3f} ± {scores_s.std():.3f}")


# plot embedding
# blue = pos, red = neg
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = ['blue' if label == 0 else 'red' for label in y]

ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
           c=colors, s=1, alpha=0.3)

ax.set_title("CEBRA 3D Embedding (Blue=Positive, Red=Negative)")

plt.savefig('cebra_embedding_3d.png', dpi=150)
plt.show()


print("Done.")
