import tensorflow as tf

def negative_log_likelih_cox(yTrue,yPred):
    # custom loss function: 
    # partial negative log likelihood of the cox proportional harzards model
    yStatus = yTrue[:,0]
    yTime = yTrue[:,1]
    
    def fun_logsumexp(i, yPred, yTime):
        j_mask = tf.greater_equal(yTime,i)
        yPred_masked = tf.boolean_mask(yPred,j_mask)
        logsumexp_ = tf.reduce_logsumexp(yPred_masked) # reduce_logsumexp?
        return logsumexp_
    
    logsumexp = tf.map_fn(lambda i: fun_logsumexp(i,yPred,yTime), yTime) 
    
    # each tensor is 1D with shape (n,)
    # after expand dim it's (n,1)
    # after concatenation it's (n,3)
    # -> enable usage tf.map_fn for computations along the n's
    yStatus = tf.expand_dims(yStatus,1)
    logsumexp = tf.expand_dims(logsumexp,1)
    d_theta_nls = tf.concat([yStatus, yPred, logsumexp], axis=1)
    
    def fun_loglikelihood_i(x):
        loss_i = x[0] * (x[1] - x[2]) # = d_i * (theta_i - logsumexp_i)
        return loss_i
    
    log_likelih_summands = tf.map_fn(lambda x: fun_loglikelihood_i(x), d_theta_nls)
    loss = -tf.reduce_sum(log_likelih_summands)
    n = tf.reduce_sum(tf.ones_like(yPred))
    batchloss = tf.div(loss,n)
    
    return batchloss 