import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    # Chunked computation to avoid memory errors on large arrays
    n = pred.shape[0]
    elem_per_sample = np.prod(pred.shape[1:]) if pred.ndim > 1 else 1
    # Use chunks if total size exceeds ~1 GiB (float32)
    if pred.nbytes > 1e9:
        chunk_size = max(1, int(1e9 / (elem_per_sample * pred.itemsize)))
        mae_sum = 0.0
        mse_sum = 0.0
        mape_sum = 0.0
        mspe_sum = 0.0
        total = 0
        for i in range(0, n, chunk_size):
            p = pred[i:i + chunk_size]
            t = true[i:i + chunk_size]
            diff = t - p
            count = diff.size
            mae_sum += np.sum(np.abs(diff))
            mse_sum += np.sum(diff ** 2)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = diff / t
                mape_sum += np.sum(np.abs(ratio))
                mspe_sum += np.sum(ratio ** 2)
            del diff, ratio
            total += count
        mae = mae_sum / total
        mse = mse_sum / total
        rmse = np.sqrt(mse)
        mape = mape_sum / total
        mspe = mspe_sum / total
    else:
        mae = MAE(pred, true)
        mse = MSE(pred, true)
        rmse = RMSE(pred, true)
        mape = MAPE(pred, true)
        mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
