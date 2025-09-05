
clear; close all; clc;
rng(1,'twister');  % reproducible

% ------------------ User settings ------------------
dataFile = 'processed.cleveland.data.txt'; % dataset (13 features + 1 label)
saveFolder = 'results';                     % where we save CSV outputs
useFixedSplit = true;                       % true = first 208 rows training, rest test
trainRows_fixed = 1:208;                    % fixed split (as in your code)
radius_percentile = 100;                    % 100 => radius = max distance (like your code)
kClusterList = 2:6;                         % clustering K's to try

% create results folder
if ~exist(saveFolder,'dir'), mkdir(saveFolder); end

% ------------------ Load data ------------------
if ~exist(dataFile,'file')
    error('Data file not found: %s', dataFile)
end
A = load(dataFile);        % expects numeric matrix
[nSamples, nCols] = size(A);
fprintf('Loaded data: %d samples, %d columns (features+label)\n', nSamples, nCols);

% ------------------ Train/test split ------------------
if useFixedSplit
    trdata = A(trainRows_fixed,:);         % first 208 rows as training
    tstdata = A(max(trainRows_fixed)+1:end,:); % rest as test (208+1 : end)
else
    perm = randperm(nSamples);
    nTrain = round(0.70 * nSamples);
    trdata = A(perm(1:nTrain),:);
    tstdata = A(perm(nTrain+1:end),:);
end
fprintf('Training samples: %d | Test samples: %d\n', size(trdata,1), size(tstdata,1));

% ------------------ z-normalization on training  -------------
% applied z-normalization feature-wise:
% z = (x - mean) / std
% Note: your original line had a small bug (missing parentheses). Fixed here.
nFeatures = 13;
trdatan = zeros(size(trdata,1), nFeatures);
means = zeros(1,nFeatures);
stds  = zeros(1,nFeatures);
for i = 1:nFeatures
    coli = trdata(:,i);
    mi = mean(coli);
    si = std(coli);
    if si == 0, si = 1e-6; end           % avoid division by zero
    zi = (coli - mi) ./ si;             % CORRECTED z-score
    trdatan(:,i) = zi;
    means(i) = mi;
    stds(i)  = si;
end

% Extract class labels from training data
cl_data = trdata(:,14);                 %  training labels (not normalized)
% At this stage the normalized training features are in trdatan (rows align with cl_data)

% ------------------ Separate class-wise training data (normalized) -----------------------
%  used class labels 0..4 — we follow that mapping and create per-class matrices.
f0 = find(cl_data == 0);
f1 = find(cl_data == 1);
f2 = find(cl_data == 2);
f3 = find(cl_data == 3);
f4 = find(cl_data == 4);

% Possibly some classes empty: handle gracefully
cl0data = trdatan(f0, :);  % normalized features for class 0
cl1data = trdatan(f1, :);
cl2data = trdatan(f2, :);
cl3data = trdatan(f3, :);
cl4data = trdatan(f4, :);

% ------------------ Hypersphere centers and radii -----------------------
c_tr0 = mean(cl0data,1); c_tr1 = mean(cl1data,1); c_tr2 = mean(cl2data,1);
c_tr3 = mean(cl3data,1); c_tr4 = mean(cl4data,1);

% Radii:  max distance of training points from center
r_tr0 = max(sqrt(sum((cl0data - c_tr0).^2, 2))); 
r_tr1 = max(sqrt(sum((cl1data - c_tr1).^2, 2))); 
r_tr2 = max(sqrt(sum((cl2data - c_tr2).^2, 2))); 
r_tr3 = max(sqrt(sum((cl3data - c_tr3).^2, 2))); 
r_tr4 = max(sqrt(sum((cl4data - c_tr4).^2, 2))); 

% If you instead prefer percentile (robust to outliers), replace the max with prctile(...)
if radius_percentile < 100
    r_tr0 = prctile(sqrt(sum((cl0data - c_tr0).^2,2)), radius_percentile);
    r_tr1 = prctile(sqrt(sum((cl1data - c_tr1).^2,2)), radius_percentile);
    r_tr2 = prctile(sqrt(sum((cl2data - c_tr2).^2,2)), radius_percentile);
    r_tr3 = prctile(sqrt(sum((cl3data - c_tr3).^2,2)), radius_percentile);
    r_tr4 = prctile(sqrt(sum((cl4data - c_tr4).^2,2)), radius_percentile);
end

% Prepare centers & radii in the same order that we will assign class labels (row index->class)
centers = [c_tr0; c_tr1; c_tr2; c_tr3; c_tr4];   % centers(1,:) => class 0
radii   = [r_tr0; r_tr1; r_tr2; r_tr3; r_tr4];

% ------------------ Prepare test set and apply same z-normalization (using saved means/stds) ---
tst_cl = tstdata(:,14);                     % true test labels
tst_features = tstdata(:, 1:13);           % raw test features
% apply the SAME means & stds computed on training set!
for i = 1:nFeatures
    tst_features(:,i) = (tst_features(:,i) - means(i)) ./ stds(i);
end

% ------------------ Hypersphere prediction using student's rules -----------------------
nTest = size(tst_features,1);
pred_hyp = zeros(nTest,1);
scores_hyp = zeros(nTest, size(centers,1));   % negative distances used as scores for AUC

tinyTol = 1e-9;  % for tie handling

for t = 1:nTest
    sample = tst_features(t, :);
    distances = sqrt(sum((centers - sample).^2, 2));    % distances to each center (rows correspond to classes 0..4)
    scores_hyp(t, :) = -distances';                     % higher score = closer (for AUC)
    in_class = distances <= (radii + tinyTol);         % logical vector: inside which spheres?
    if sum(in_class) == 1
        idx = find(in_class == 1);        % index of class (1..5)
        pred_hyp(t) = idx - 1;            % label mapping: index-1 => class label 0..4
    elseif sum(in_class) > 1
        % Multiple spheres contain sample -> uncertain (-1)
        pred_hyp(t) = -1;
    else
        % Not inside any sphere -> unclassified (-2)
        pred_hyp(t) = -2;
    end
end

% ------------------ Evaluate Hypersphere predictions ------------------------
confident_idx = pred_hyp >= 0;
if sum(confident_idx) == 0
    confident_accuracy = NaN;
else
    correct = pred_hyp(confident_idx) == tst_cl(confident_idx);
    confident_accuracy = sum(correct) / sum(confident_idx);
end
overall_accuracy_all = mean(pred_hyp == tst_cl);  % counts -1/-2 as incorrect (unless equal by chance)
fprintf('Hypersphere: Confident accuracy: %.2f%% (confident samples = %d / %d)\n', ...
    confident_accuracy*100, sum(confident_idx), nTest);
fprintf('Hypersphere: Overall accuracy (treat -1/-2 as wrong): %.2f%%\n', overall_accuracy_all*100);
fprintf('Hypersphere: Uncertain = %d | Unclassified = %d\n', sum(pred_hyp==-1), sum(pred_hyp==-2));

% ------------------ Build tr_param for Bayes ----------------------
% tr_param is d x 2 x numClasses
classes = 0:4;
tr_param = build_tr_param_from_training_data(cl0data, cl1data, cl2data, cl3data, cl4data);
% compute class priors from training set counts
Ntotal_train = size(trdatan,1);
cls_prior = [numel(f0), numel(f1), numel(f2), numel(f3), numel(f4)] ./ Ntotal_train;

% ------------------ Bayes predictions ---------------------
pred_bayes = zeros(nTest,1);
scores_bayes = zeros(nTest, numel(classes)); % will hold posterior probabilities
for t = 1:nTest
    x = tst_features(t,:);
    [cls_out, prob1] = test_bayes(x, tr_param, cls_prior);
    pred_bayes(t) = cls_out;         % uses 0..4 labels
    scores_bayes(t, :) = prob1(:)';  % posterior vector
end

% Evaluate Bayesian predictions
overall_acc_bayes = mean(pred_bayes == tst_cl);
fprintf('Bayes (student): Overall accuracy = %.2f%%\n', overall_acc_bayes*100);

% ------------------ Gaussian Naive Bayes (separate) -----------------------------
[gnb_model] = train_gnb_from_trdatan(trdatan, cl_data);  % trdatan = normalized training features
[pred_gnb, scores_gnb] = predict_gnb(gnb_model, tst_features);
overall_acc_gnb = mean(pred_gnb == tst_cl);
fprintf('Gaussian Naive Bayes: Overall accuracy = %.2f%%\n', overall_acc_gnb*100);

% ------------------ 1-NN (simple) -------------------------------------------------
[pred_nn, scores_nn] = predict_1nn_and_scores(trdatan, cl_data, tst_features);
overall_acc_nn = mean(pred_nn == tst_cl);
fprintf('1-NN: Overall accuracy = %.2f%%\n', overall_acc_nn*100);

% ------------------ Compute per-class metrics and save CSVs ----------------------
metrics_hyp = evaluate_multiclass(tst_cl, pred_hyp, scores_hyp, classes);
metrics_bayes = evaluate_multiclass(tst_cl, pred_bayes, scores_bayes, classes);
metrics_gnb   = evaluate_multiclass(tst_cl, pred_gnb, scores_gnb, classes);
metrics_nn    = evaluate_multiclass(tst_cl, pred_nn, scores_nn, classes);

% Save per-class tables to CSV for uploading to GitHub
writetable(struct2table_row(metrics_hyp,'hypersphere','raw'), fullfile(saveFolder,'perclass_hyp_raw.csv'));
writetable(struct2table_row(metrics_bayes,'bayes','raw'), fullfile(saveFolder,'perclass_bayes_raw.csv'));
writetable(struct2table_row(metrics_gnb,'gnb','raw'), fullfile(saveFolder,'perclass_gnb_raw.csv'));
writetable(struct2table_row(metrics_nn,'1nn','raw'), fullfile(saveFolder,'perclass_1nn_raw.csv'));

% Summary CSV
summaryTbl = table({'Hypersphere'; 'Bayes(student)'; 'GaussianNB'; '1-NN'}, ...
    [confident_accuracy; overall_acc_bayes; overall_acc_gnb; overall_acc_nn], ...
    'VariableNames', {'Model', 'Accuracy'});
writetable(summaryTbl, fullfile(saveFolder,'results_summary.csv'));
fprintf('Saved per-class CSVs and results_summary.csv to folder "%s"\n', saveFolder);

% ------------------ Clustering and Davies-Bouldin index -------------------------
Xtrain_for_clust = trdatan;  % normalized training features (student used trdatan)
[kmeans_info, hier_info] = run_clustering_and_db(Xtrain_for_clust, kClusterList);
% Save DB values into CSV for quick view
kvals = kClusterList';
kmeansDBs = arrayfun(@(i) kmeans_info(i).DB, 1:numel(kClusterList))';
hierDBs   = arrayfun(@(i) hier_info(i).DB, 1:numel(kClusterList))';
clustTbl = table(kvals, kmeansDBs, hierDBs, 'VariableNames', {'K','Kmeans_DB','Hier_DB'});
writetable(clustTbl, fullfile(saveFolder,'clustering_db.csv'));
fprintf('Saved clustering DB indices to %s\\n', fullfile(saveFolder,'clustering_db.csv'));

% ------------------ Done -------------------------------------------------------
fprintf('\\nPROJECT COMPLETE — check the results/ folder for CSVs.\\n');

%% ---------------- Local helper functions ----------------
% These are functions with small fixes and safety checks
% so they integrate cleanly with the project. I kept names and logic to make it familiar.

function p = gaussian_dis(x,m,s)
    % student's gaussian PDF for one scalar x, mean m and std s
    if s == 0, s = 1e-6; end
    a = -0.5 * ((x - m)/s)^2;
    b = exp(a);
    p = (1 / (sqrt(2*pi) * s)) * b;
end

function [cls_probj,likelihood1] = posterior1(x,tr_paramj,prob_clsj)
    % posterior1: compute (unnormalized) posterior numerator for one class
    % x: 1xd feature vector
    % tr_paramj: d x 2 matrix [mu sigma] per feature for this class
    % prob_clsj: class prior (scalar)
    likelihood1 = 1;
    for ii = 1:length(x)
        pi = gaussian_dis(x(ii), tr_paramj(ii,1), tr_paramj(ii,2));
        if pi == 0, pi = 1e-6; end
        likelihood1 = likelihood1 * pi;
    end
    cls_probj = likelihood1 * prob_clsj; % numerator of Bayes rule
end

function [cls_out,prob1] = test_bayes(x,tr_param,cls_prob)
    % test_bayes: wrapper that computes posterior for each class and normalizes
    nClasses = size(tr_param,3);
    prob1 = zeros(1,nClasses);
    for i = 1:nClasses
        [prob1(i), ~] = posterior1(x, tr_param(:,:,i), cls_prob(i));
    end
    if sum(prob1) == 0
        prob1 = ones(size(prob1)) / numel(prob1);
    else
        prob1 = prob1 / sum(prob1);  % normalize to sum=1 (posterior)
    end
    ind_max_prob = find(prob1 == max(prob1));
    ind_max_prob = min(ind_max_prob); % resolve ties by taking smallest index
    cls_out = ind_max_prob - 1;       % student's label mapping (1->0, 2->1, ...)
end

function tr_param = build_tr_param_from_training_data(cl0data,cl1data,cl2data,cl3data,cl4data)
    % builds tr_param = d x 2 x 5 where for each class we have [mu sigma] per feature
    d = size(cl0data,2);
    tr_param = zeros(d,2,5);
    % class 0
    if isempty(cl0data), tr_param(:,:,1) = repmat([0 1],d,1); else tr_param(:,:,1) = [mean(cl0data)' std(cl0data,0,1)']; end
    if isempty(cl1data), tr_param(:,:,2) = repmat([0 1],d,1); else tr_param(:,:,2) = [mean(cl1data)' std(cl1data,0,1)']; end
    if isempty(cl2data), tr_param(:,:,3) = repmat([0 1],d,1); else tr_param(:,:,3) = [mean(cl2data)' std(cl2data,0,1)']; end
    if isempty(cl3data), tr_param(:,:,4) = repmat([0 1],d,1); else tr_param(:,:,4) = [mean(cl3data)' std(cl3data,0,1)']; end
    if isempty(cl4data), tr_param(:,:,5) = repmat([0 1],d,1); else tr_param(:,:,5) = [mean(cl4data)' std(cl4data,0,1)']; end
    % avoid zero std
    for c = 1:5
       tmp = tr_param(:,2,c);
       tmp(tmp == 0) = 1e-6;
       tr_param(:,2,c) = tmp;
    end

end

function model = train_gnb_from_trdatan(trdatan, cl_data)
    % Train Gaussian Naive Bayes using normalized training data (trdatan)
    classes = unique(cl_data);
    d = size(trdatan,2);
    model.classes = classes;
    for i = 1:numel(classes)
        lab = classes(i);
        Xc = trdatan(cl_data == lab, :);
        mu = mean(Xc,1)'; sigma = std(Xc,0,1)';
        sigma(sigma == 0) = 1e-6;
        model.params(:,:,i) = [mu sigma];
        model.prior(i) = size(Xc,1)/size(trdatan,1);
    end
end

function [preds, scores] = predict_gnb(model, Xtest)
    % Predict using Gaussian Naive Bayes modeled in train_gnb_from_trdatan
    C = numel(model.classes); n = size(Xtest,1);
    preds = zeros(n,1); scores = zeros(n,C);
    for t = 1:n
        x = Xtest(t,:)';
        post = zeros(1,C);
        for ci = 1:C
            mu = model.params(:,1,ci); sigma = model.params(:,2,ci);
            % compute log-likelihood to avoid underflow
            logpdf = sum(-0.5 * ((x - mu)./sigma).^2 - log(sqrt(2*pi)*sigma));
            post(ci) = exp(logpdf) * model.prior(ci);
        end
        if sum(post) == 0, post = ones(size(post))/numel(post); else post = post / sum(post); end
        [~, idx] = max(post); preds(t) = model.classes(idx); scores(t,:) = post;
    end
end

function [preds, scores] = predict_1nn_and_scores(Xtrain, ytrain, Xtest)
    % 1-NN predictions + simple per-class score = -min distance to any train sample of that class
    n = size(Xtest,1); classes = unique(ytrain); C = numel(classes);
    preds = zeros(n,1); scores = zeros(n,C);
    for t = 1:n
        x = Xtest(t,:);
        D = sqrt(sum((Xtrain - x).^2, 2));
        [~, idx] = min(D);
        preds(t) = ytrain(idx);
        for ci = 1:C
            mask = (ytrain == classes(ci));
            if any(mask)
                dmin = min(sqrt(sum((Xtrain(mask,:) - x).^2,2)));
                scores(t,ci) = -dmin;
            else
                scores(t,ci) = -Inf;
            end
        end
    end
end

function metrics = evaluate_multiclass(ytrue, ypred, scores, classes)
    % Computes confusion matrix, TP,FP,FN,TN, sensitivity, specificity, MCC, AUC (one-vs-rest)
    N = numel(ytrue); C = numel(classes);
    metrics.classes = classes(:)';
    metrics.accuracy = mean(ypred == ytrue);
    conf = zeros(C,C);
    for i=1:N
        t = ytrue(i); p = ypred(i);
        rid = find(classes == t); cid = find(classes == p);
        if ~isempty(rid) && ~isempty(cid), conf(rid,cid) = conf(rid,cid) + 1; end
    end
    metrics.confusionMatrix = conf;
    metrics.TP = zeros(C,1); metrics.FP = zeros(C,1); metrics.FN = zeros(C,1); metrics.TN = zeros(C,1);
    for ci = 1:C
        c = classes(ci);
        TP = sum((ytrue==c) & (ypred==c));
        FP = sum((ytrue~=c) & (ypred==c));
        FN = sum((ytrue==c) & (ypred~=c));
        TN = N - TP - FP - FN;
        metrics.TP(ci)=TP; metrics.FP(ci)=FP; metrics.FN(ci)=FN; metrics.TN(ci)=TN;
        metrics.sensitivity(ci) = safe_div(TP, TP+FN);
        metrics.specificity(ci) = safe_div(TN, TN+FP);
        denom = sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
        if denom==0, metrics.MCC(ci)=NaN; else metrics.MCC(ci)=(TP*TN - FP*FN)/denom; end
    end
    % AUC per class using scores (higher = more likely); if perfcurve unavailable, fallback to ranks
    metrics.AUC = zeros(C,1);
    for ci=1:C
        posmask = (ytrue == classes(ci));
        sc = scores(:,ci);
        try
            [~,~,~,A] = perfcurve(posmask, sc, true);
            metrics.AUC(ci) = A;
        catch
            metrics.AUC(ci) = auc_rank(posmask, sc);
        end
    end
    metrics.AUC_macro = mean(metrics.AUC(~isnan(metrics.AUC)));
end

function v = safe_div(a,b)
    if b==0, v = NaN; else v = a/b; end
end

function A = auc_rank(posmask, scores)
    % Rank-based AUC (Mann-Whitney U) fallback
    pos = scores(posmask); neg = scores(~posmask);
    npos = numel(pos); nneg = numel(neg);
    if npos==0 || nneg==0, A = NaN; return; end
    allS = [pos; neg];
    [sorted, idx] = sort(allS);
    ranks = zeros(size(allS));
    i = 1; N = numel(allS);
    while i <= N
        j = i;
        while j < N && sorted(j+1) == sorted(i), j = j + 1; end
        avg = mean(i:j);
        ranks(idx(i:j)) = avg;
        i = j + 1;
    end
    sumRanksPos = sum(ranks(1:npos));
    U = sumRanksPos - npos*(npos+1)/2;
    A = U / (npos * nneg);
end

function T = struct2table_row(metrics, methodName, mode)
    % Ensure everything is column vector of same length
    C = metrics.classes(:);
    TP = metrics.TP(:);
    FP = metrics.FP(:);
    FN = metrics.FN(:);
    TN = metrics.TN(:);
    Sens = metrics.sensitivity(:);
    Spec = metrics.specificity(:);
    MCC = metrics.MCC(:);
    AUC = metrics.AUC(:);

    % Build table row-wise
    T = table(C, TP, FP, FN, TN, Sens, Spec, MCC, AUC, ...
        'VariableNames', {'Class','TP','FP','FN','TN','Sensitivity','Specificity','MCC','AUC'});
    
    % Store description (optional)
    T.Properties.Description = sprintf('%s (mode=%s)', methodName, mode);
end


function [kmeans_info, hier_info] = run_clustering_and_db(X, kList)
    % Run kmeans and hierarchical clustering for k in kList and compute Davies-Bouldin index
    kmeans_info = struct('idx',[],'centroids',[],'DB',[]);
    hier_info = struct('idx',[],'centroids',[],'DB',[]);
    for i=1:numel(kList)
        k = kList(i);
        [idx,C] = kmeans(X, k, 'Replicates',5, 'MaxIter',500);
        db = davies_bouldin_index(X, idx, C);
        kmeans_info(i).idx = idx; kmeans_info(i).centroids = C; kmeans_info(i).DB = db;
        Z = linkage(X,'ward');
        idx2 = cluster(Z,'maxclust',k);
        C2 = zeros(k, size(X,2));
        for c=1:k, C2(c,:) = mean(X(idx2==c,:),1); end
        db2 = davies_bouldin_index(X, idx2, C2);
        hier_info(i).idx = idx2; hier_info(i).centroids = C2; hier_info(i).DB = db2;
    end
end

function db = davies_bouldin_index(X, labels, centroids)
    k = size(centroids,1);
    S = zeros(k,1);
    for i=1:k
        Xi = X(labels==i,:);
        if isempty(Xi), S(i)=0; else S(i) = mean(sqrt(sum((Xi - centroids(i,:)).^2,2))); end
    end
    M = pdist2(centroids, centroids);
    R = zeros(k,k);
    for i=1:k, for j=1:k
            if i == j, R(i,j)=NaN; else R(i,j) = (S(i) + S(j)) / (M(i,j) + eps); end
        end
    end
    Rmax = max(R,[],2);
    db = mean(Rmax);
end
