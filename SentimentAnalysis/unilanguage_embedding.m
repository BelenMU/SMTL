%% Unilingual embedding pre-processing
% Preprocess monolingual word embeddings cured by Artetxe
% OUTPUT: "initialization.mat" a file with all the normalized word
% embeddings in Spanish and Italian, as well as the classifiers for good
% vs. bad words.
%% Get embeddings
% Prerequisit, download datasets from http://ixa2.si.ehu.es/martetxe/vecmap/es.emb.txt.gz
embeddings_it = dlmread("it.emb.txt", ' ', 1, 1); % It only reads numbers
embeddings_es = dlmread("es.emb.txt", ' ', 1, 1); % It only reads numbers
%% Words in Spanish
fid = fopen("es.emb.txt");
tline = fgetl(fid);
tline = fgetl(fid);
words_es = {};
while ischar(tline)
    temp = split(tline);
    words_es = [words_es; temp{1}];
    tline = fgetl(fid);
end
%% Words in Italian
fid = fopen("it.emb.txt");
tline = fgetl(fid);
tline = fgetl(fid);
words_it = {};
while ischar(tline)
    temp = split(tline);
    words_it = [words_it; temp{1}];
    tline = fgetl(fid);
end

%% Normalize Embedding as suggested by \cite{Xing2015}
embeddings_es = embeddings_es';
embeddings_en = embeddings_en';
embeddings_it = embeddings_it';
emb_norm_es = embeddings_es ./ vecnorm(embeddings_es);
emb_norm_it = embeddings_it ./ vecnorm(embeddings_it);
emb_norm_en = embeddings_en ./ vecnorm(embeddings_en);
%% Find Classifier in Spanish Positive vs. Negative
% Words from https://www.kaggle.com/datasets/rtatman/sentiment-lexicons-for-81-languages
% Find Positive Adjectives Examples
pos_words_es = readtable("positive_words_es.txt");
pos_words_es = table2cell(pos_words_es);
pos_ind_es = [];
for ii = 1:length(pos_words_es)
    temp = strfind(words_es, pos_words_es{ii});
    pos_ind_es = [pos_ind_es; find(not(cellfun('isempty',temp)))];
end
% Find Negative Adjectives Examples
neg_words_es = readtable("negative_words_es.txt");
neg_words_es = table2cell(neg_words_es);
neg_ind_es = [];
for ii = 1:length(neg_words_es)
    temp = strfind(words_es, neg_words_es{ii});
    neg_ind_es = [neg_ind_es; find(not(cellfun('isempty',temp)))];
end
% Create Training Examples
len = min(length(pos_ind_es), length(neg_ind_es));% So that dataset is balanced
i_pos = randsample(length(pos_ind_es), len);
i_neg = randsample(length(neg_ind_es), len);
x_train = [ones(1, 2*len); emb_norm_es(:, pos_ind_es(i_pos)), emb_norm_es(:, neg_ind_es(i_neg))];
y_train = [ones(len, 1); -1*ones(len, 1)];
% Theta
theta = lsqlin(x_train', y_train);
theta = theta ./ norm(theta);


%% Find Classifier in Italian Positive vs. Negative
% Find Positive Adjectives Examples
pos_words_it = readtable("positive_words_it.txt");
pos_words_it = table2cell(pos_words_it);
pos_ind_it = [];
for ii = 1:length(pos_words_it)
    temp = strfind(words_it, pos_words_it{ii});
    pos_ind_it = [pos_ind_it; find(not(cellfun('isempty',temp)))];
end
% Find Negative Adjectives Examples
neg_words_it = readtable("negative_words_it.txt");
neg_words_it = table2cell(neg_words_it);
neg_ind_it = [];
for ii = 1:length(neg_words_it)
    temp = strfind(words_it, neg_words_it{ii});
    neg_ind_it = [neg_ind_it; find(not(cellfun('isempty',temp)))];
end
% Create Training Examples
len = min(length(pos_ind_it), length(neg_ind_it));% So that dataset is balanced
i_pos = randsample(length(pos_ind_it), len);
i_neg = randsample(length(neg_ind_it), len);
x_test_it = [ones(1, 2*len); emb_norm_it(:, pos_ind_it(i_pos)), emb_norm_it(:, neg_ind_it(i_neg))];
y_test_it = [ones(len, 1); -1*ones(len, 1)];
% Theta
theta_it = lsqlin(x_test_it', y_test_it);
theta_it = theta_it ./ norm(theta_it);

%% Save only 10000 most common words
emb_norm_es_commun = emb_norm_es(:, 1:1e4); 
words_es_commun = words_es_commun(1:1e4);

save("initialization.mat")