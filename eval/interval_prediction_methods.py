import torch

def greedy_coalescing(probs, thresholds, k=1):
    """
    params-
    probs: [n x t]- tensor, prob score for each time (t times) for each instance
    thresholds: [n x 1]- threshold for each instance
    k: number of intervals to be returned for each
    returns-
    (start, end), where start and end are [n x k] tensors,
    indicating boundaries for top k intervals for each instance
    """

    batch_size = len(probs)
    num_times = probs.shape[-1]

    # _,best_indices= torch.max(probs,-1)
    indices = torch.argsort(probs, descending=True)[:, :k]

    pred_min = torch.zeros(batch_size, k)
    pred_max = torch.zeros(batch_size, k)

    for i in range(batch_size):
        for idx, best_t in enumerate(indices[i]):
            best_t = int(best_t)
            left, right = best_t, best_t
            tot = probs[i, best_t]

            thresh = thresholds[i]

            # print("\ntot:{}, thresh:{}".format(tot,thresh))

            # print("Initial:",left,right)
            while tot < thresh and (left > 0 or right < num_times - 1):
                next_index = -1
                if (left == 0):
                    right += 1
                    next_index = right
                elif (right == num_times - 1):
                    left -= 1
                    next_index = left
                else:
                    left_score = probs[i, left - 1]
                    right_score = probs[i, right + 1]
                    # print("left_score:{}, right_score:{}".format(left_score,right_score))

                    if (left_score > right_score):
                        left -= 1
                        next_index = left
                    else:
                        right += 1
                        next_index = right

                tot += probs[i, next_index]

            # print("pred_min shape:",pred_min.shape)
            # print("indices shape:",indices.shape)
            pred_min[i, idx] = left
            pred_max[i, idx] = right

        # print("Later:",left,right)

    # print("{}/{} done".format(i,batch_size))

    return pred_min, pred_max
