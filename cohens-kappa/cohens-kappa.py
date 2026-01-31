def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    n = len(rater1)
    agreements = 0
    unique_labels1 = {}
    unique_labels2 = {}
    unique_labels = set()

    for ind in range(n):
        if rater1[ind] == rater2[ind]:
            agreements += 1
        unique_labels.add(rater1[ind])
        unique_labels.add(rater2[ind])
        unique_labels1[rater1[ind]] = unique_labels1.get(rater1[ind], 0) + 1
        unique_labels2[rater2[ind]] = unique_labels2.get(rater2[ind], 0) + 1
    
    po = agreements / n
    
    pe = 0
    for k in unique_labels:
        pe += (
            (unique_labels1.get(k, 0)/n) * (unique_labels2.get(k, 0)/n)
        )
    if pe == 1.0:
        return 1.0
    
    kappa = (po - pe) / (1 - pe)

    return kappa
    
    