import casadi

class model:

    def __init__(self, ylist, theta, alpha, beta, x):
        
        self.y=y
        self.theta=theta
        self.alpha=alpha
        self.beta=beta
        self.x=x

        for ytuple in self.ylist

                 switch dist
                     case 'Norm'
                         
                        #set logliklihood symbolics and function
                        obj.pdf(i)= 1/sqrt(2*pi*obj.mu(2)) * exp(-(obj.y-obj.mu(1))^2./(2*obj.mu(2)));
                        obj.pdf_func(i) = Function('pdf_func', {obj.y,obj.theta,obj.u}, {obj.pdf{i}});
                        #set logliklihood symbolics and function
                        obj.loglik(i) = log(obj.pdf);#%-0.5*log(2*pi*obj.mu(2)) - (obj.x-obj.mu(1))^2/(2*obj.mu(2));
                        obj.loglik_func(i) = Function('loglik', {obj.y,obj.theta,obj.u}, {obj.loglik{i}});
                        #set FIM symbolics and function
                        mu_theta=jacobian(obj.mu,obj.theta);
                        obj.fim(i)=(mu_theta(1,:)'*mu_theta(1,:))./obj.mu(2)+(mu_theta(2,:)'*mu_theta(2,:))./obj.mu(2)^2;
                        obj.fim_func(i)=Function('fim', {obj.theta,obj.u}, {obj.fim{i}});






