clear variables


% load training and test data
load('mat_data/train.mat')
load('mat_data/test.mat')


% show the first 5 images in training data
for i = 1:5
    subplot(1,5,i)
    I = reshape(Ytrain(:,i),28,23);
    imagesc(I);
    colormap(gray);
    axis image;
    set(gca,'xtick',[],'ytick',[])
end


% show the first 5 images in test data
for i = 1:5
    subplot(1,5,i)
    I = reshape(Ytest(:,i),28,23);
    imagesc(I);
    colormap(gray);
    axis image;
    set(gca,'xtick',[],'ytick',[])
end

