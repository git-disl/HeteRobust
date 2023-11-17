import numpy as np

_allFocalDiversityMetrics = set([
    'pair', # for pairwise BD
    'nonpair', # for nonpairwise GD
])


def normalize01(array):
    if max(array) == min(array): #TODO: to consider more cases
        return array
    return (array-min(array))/(max(array)-min(array))


def getNTeamStatisticsTeamName(teamNameList, accuracyDict, minAcc, avgAcc, maxAcc, tmpAccList):
    nAboveMin = 0
    nAboveAvg = 0
    nAboveMax = 0
    nHigherMember = 0
    allAcc = []
    for teamName in teamNameList:
        acc = accuracyDict[teamName]
        allAcc.append(acc)
        if acc >= round(minAcc, 2):
            nAboveMin += 1
        if acc >= round(avgAcc, 2):
            nAboveAvg += 1
        if acc >= round(maxAcc, 2):
            nAboveMax += 1
            #print(teamName)
        # count whether an ensemble is higher than all its member model
        nHigherMember += 1
        for modelName in teamName.split(','):
            modelAcc = tmpAccList[int(modelName)]
            if acc < modelAcc:
                nHigherMember -= 1
                break
    return len(teamNameList), np.min(allAcc), np.max(allAcc), np.mean(allAcc), np.std(allAcc), nHigherMember, nAboveMax, nAboveAvg, nAboveMin


def getTopNTeamStatistics(teamNameList, accuracyDict, diversityScoreDict, dm=None, topN=10):
    if dm:
        diversityMAPTeamNameList = [(diversityScoreDict[teamName][dm], accuracyDict[teamName], teamName) for teamName in teamNameList]
    else:
        diversityMAPTeamNameList = [(diversityScoreDict[teamName], accuracyDict[teamName], teamName) for teamName in teamNameList]
    
    diversityMAPTeamNameList.sort(reverse=True)
    
    topNmAP = []
    for i in range(topN):
        topNmAP.append((diversityMAPTeamNameList[i][1], diversityMAPTeamNameList[i][0], diversityMAPTeamNameList[i][2]))
    
    print(min(topNmAP), max(topNmAP))


# focal negative correlations
# https://github.com/pdoren/DeepEnsemble/deepensemble/metrics/diversitymetrics.py
# https://github.com/git-disl/EnsembleBench/blob/main/EnsembleBench/diversityMetrics.py
eps = 0.0000001
def generalized_diversity(M): # nonpairwise
    N = M.shape[1] # num samples
    L = M.shape[0] # num models
    pi = []
    fail = L - np.sum(M, axis=0)
    for i in range(L+1):
        ff = np.sum(fail == i)
        pi.append(np.mean(ff))
        
    pi = pi / np.sum(pi)
    
    P1 = 0.0
    P2 = 0.0
    for i in range(1, L+1):
        P1 += i * 1.0 * pi[i] / L
        P2 += i * (i-1) * 1.0 * pi[i] / (L * (L-1))
    
    if P1 == P2 and P1 == 0.0:
        return 0.0
    
    return 1.0-P2/(P1 + eps)


def binary_disagreement(M): # pairwise
    Qs = []
    for i in range(M.shape[0]):
        for j in range(i + 1, M.shape[0]):
            N_1_1 = np.sum(np.logical_and(M[i, :], M[j, :]))  # number of both correct
            N_0_0 = np.sum(np.logical_and(np.logical_not(M[i, :]), np.logical_not(M[j, :])))  # number of both incorrect
            N_0_1 = np.sum(np.logical_and(np.logical_not(M[i, :]), M[j, :]))  # number of j correct but not i
            N_1_0 = np.sum(np.logical_and(M[i, :], np.logical_not(M[j, :])))  # number of i correct but not j
            Qs.append((N_0_1+N_1_0)*1./(N_1_1+N_1_0+N_0_1+N_0_0))
    return np.mean(Qs)


def calDiversityHelper(predResults, metrics=None):
    '''
    predResults: LxN, L: number of models, N: number of detections (0: correct, 1: wrong)
    metrics: check _allFocalDiversityMetrics
    '''
    if metrics == None:
        return
    results = list()
    for m in metrics:
        #print(m)
        if m not in _allFocalDiversityMetrics:
            raise Exception("Focal Negative Correlations Not Found!")
        elif m == "pair":
            results.append(binary_disagreement(predResults))
        elif m == "nonpair":
            results.append(generalized_diversity(predResults))
    return results

def calFocalNegativeCorrelations(
    oneFocalModel,
    teamModelList,
    allPredictions,
    negative_sample_list,
    diversityMetricsList,
    # save time
    crossValidation = True,
    nRandomSamples = 100,
    crossValidationTimes = 3,
    randomSeed = 2021
):
    np.random.seed(randomSeed) # fix random state
    teamPredictions = allPredictions[teamModelList]
    if crossValidation:
        tmpMetrics = list()
        for _ in range(crossValidationTimes):
            randomIdx = np.random.choice(np.arange(len(negative_sample_list[oneFocalModel])), nRandomSamples)        
            tmpMetrics.append(np.array(calDiversityHelper(
                              teamPredictions[..., negative_sample_list[oneFocalModel][randomIdx]],
                              diversityMetricsList)))
        tmpMetrics = np.mean(np.array(tmpMetrics), axis=0)
    else:
        tmpMetrics = np.array(calDiversityHelper(
                              teamPredictions[..., negative_sample_list[oneFocalModel]],
                              diversityMetricsList))

    return {diversityMetricsList[i]:tmpMetrics[i].item()  for i in range(len(tmpMetrics))}


if __name__ == "__main__":
    # test
    M = np.array([
        [1, 1],
        [1, 0],
    ])
    print(binary_disagreement(M))

