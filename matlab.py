"""xk = x0;
    tic;
    for iter = [1: maxiters]
        [f,grad] = fhandle(xk);
        gradnorms(iter) = norm(grad);
        if gradnorms(iter)/gradnorms(1) < tolgradnorm || toc > maxtime
            break
        end
        if(iter ~= 1)
            alpha = alpha * gradnorms(iter-1)/gradnorms(iter);
        end
        while alpha > alphamin && fhandle(xk - alpha*grad) > f - c*alpha*grad'*grad
            alpha = rho*alpha;
        end
        xk = xk - max(alpha,alphamin) * grad/gradnorms(iter);
        times(iter) = toc;
    end
xk = x0 """
def f(maxiters, gradient, gamma):
    for iter in range(0, maxiters-1):
        grad = gradient(xk)
        if False:
            break
        xk = xk - gamma * grad
    return xk