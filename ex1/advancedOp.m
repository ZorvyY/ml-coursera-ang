function [optTheta] = advancedOp(X, y)

  options = optimset('GradObj', 'on', 'MaxIter', 100);
  initialTheta = zeros(2,1);

  [optTheta, functionVal, exitFlag] = fminunc(@(theta) costFunction(X, y, theta), initialTheta, options);
  %[optTheta, functionVal, exitFlag] = fminunc(
  %  @(theta) deal(computeCostMulti(X, y, theta), (1/m)*(X'*(X*theta-y))), 
  %  initialTheta, 
  %  options);

end

function [J, grad] = costFunction(X, y, theta)

  J = computeCostMulti(X, y, theta);
  grad = 1/length(y)*X'*(X*theta-y);

end
