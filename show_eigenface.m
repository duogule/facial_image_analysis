clear variables

% load training and test data
load('mat_data/train.mat')
load('mat_data/test.mat')

% compute Covariance matrix of features in training data
C = cov(Ytrain');

% compute the singular value decomposition (SVD) of the Covariance matrix 
[U, S, V] = svd(C);

% plot first 10 principal components (eigen-faces)
for i = 1:10
    subplot(2,5,i)
    I = reshape(U(:,i),28,23);
    imagesc(I);
    colormap(gray);
    axis image;
    set(gca,'xtick',[],'ytick',[]) % remove x and y axis units
end