import numpy as np
from scipy import integrate
from scipy.misc import derivative
from matplotlib import pyplot as plt


def logistic(x, center, scale):
    return (1 + np.exp(-scale * (x - center)))**(-1)


def rbf(x, center, scale):
    return np.exp(-scale * (x - center)**2)


def rbf_prime(x, center, scale):
    """derivative of radical basis functions"""
    return 2 * scale * (center - x) * rbf(x, center, scale)

for a in range(51):
    a = a / 5688666568851
    #print(a)
    #print(max(np.roots([-a, -a, 3, 4, 1])))
    #plt.plot(a, max(np.roots([-a, -a, 3, 4, 1])), "ro")

#plt.show()
# My functions to approximate are:
x = np.linspace(-1, 2, 4000)
#plt.plot(x, [rbf(i, 2, 1) for i in x], label="rbf")
#plt.plot(x, [logistic(i, 2, 2) for i in x], label="logistic")
#plt.plot(x, [logistic(i, 2, 2)*logistic(i, 1, 2) for i in x], label="logistic^2")
#diff = [logistic(i, 2, 2) - logistic(i, 2, 2)*logistic(i, 1, 2) for i in x]
#print(max(diff), logistic(.9, 2, 2) - logistic(.9, 2, 2)**2)
#print(diff.index(max(diff)))
#plt.plot(x, [logistic(i, 2, 1)*-0.1 + 0.1 + .2*(rbf(i, 2, 1)) for i in x], label="logistic derivative approx")
#plt.plot(x, [rbf(i, 2, 1)**2 * logistic(i, 2, 1)**2 for i in x], label="logistic derivative actual")
#plt.plot(x, [rbf(i, 2, 1)**2 * logistic(i, 2.1, 1)**2 for i in x], label="logistic derivative actual")
#plt.plot(x, [logistic(i, 2, 1)*-0.1 + 0.1 + .95*(rbf(i, 2, 1)) for i in x], label="RBF derivative approx")
#plt.plot(x, [rbf(i, 2, 1)**2 for i in x], label="RBF derivative actual")
#plt.plot(x, [rbf(i, 2.1, 1)**2 for i in x], label="RBF derivative actual")
# Plot as epsilon varies:
eps = np.linspace(0, 1, 2000)
epsilon = 1
alpha = 20
g = 1/(3.208 * alpha)
my_array = [logistic(i, epsilon, alpha) * logistic(i, 0, alpha) for i in x]


f, (p1, p2, p3) = plt.subplots(3, sharex=True, sharey=True)
#p1.set_title("Products of Sigmoids")
p1.plot(x, [logistic(i, epsilon, alpha) for i in x], label="Centered at {0}".format(str(epsilon)))
p1.legend()
p2.plot(x, [logistic(i, 0, alpha) for i in x], label="Centered at 0")
p2.legend(loc="lower right")
p3.plot(x, my_array, label="Product of the two.")
#plt.plot(0, -logistic(0, 0, alpha)**3 + logistic(0, 0, alpha), "ro")
#plt.plot(g, -logistic(g, 0, alpha)**3 + logistic(g, 0, alpha), "bo")
print(0, -logistic(0, 0, alpha)**3 + logistic(0, 0, alpha), g, -logistic(g, 0, alpha)**3 + logistic(g, 0, alpha))
print(x[my_array.index(max(my_array))], max(my_array))


#plt.plot(eps, [1 / (2 * (1 + np.exp(1 * e))) for e in eps], label="x=0")
#plt.plot(eps, [np.exp(-1 * e) / (2 * (1 + np.exp(-1 * e))) for e in eps], label="x=epsilon")
#outs = [-logistic(i, epsilon, alpha)*logistic(i, 0, alpha)**2 + logistic(i, epsilon, alpha) for i in x]
#plt.plot(x, outs, label="x=diff")
diffs2 = [np.abs((2*np.exp(-e/4)+ np.exp(-e/2))/(1+2*np.exp(-e/4)+ np.exp(-e/2) + np.exp(3*e/4)+ 2*np.exp(e/2)+ np.exp(e/4))) -
           np.abs((2*np.exp(-e)+ np.exp(-2*e))/(1+4*np.exp(-e)+ 2*np.exp(-2*e) + np.exp(0))) for e in eps]
#plt.plot(eps, diffs2)
#plt.plot(epsilon, -(logistic(epsilon, epsilon, alpha)*logistic(epsilon, 0, alpha)**2 - logistic(epsilon, epsilon, alpha)), "bo")
my_poly = [-2 * np.exp(-alpha * epsilon), -2 * np.exp(-alpha * epsilon), 3, 4, 1]
xs = [np.log((y) ** (1 / alpha)) for y in np.roots(my_poly)]
#print(xs)
#print("y^alpha is ", my_poly[0])
mv = xs[0]
#plt.plot(mv, -(logistic(mv, epsilon, alpha)*logistic(mv, 0, alpha)**2 - logistic(mv, epsilon, alpha)), "ro")
plt.legend()
i = xs[0]
#print(i, x[outs.index(max(outs))], max(outs),  -logistic(i, epsilon, alpha)*logistic(i, 0, alpha)**2 + logistic(i, epsilon, alpha))
of_mv = logistic(mv, epsilon, alpha)*logistic(mv, 0, alpha)**2 - logistic(mv, epsilon, alpha)
of_eps = logistic(epsilon, epsilon, alpha)*logistic(epsilon, 0, alpha)**2 - logistic(epsilon, epsilon, alpha)
#print(mv, -of_mv, "epsilon", -of_eps, "difference", of_eps-of_mv)

plt.show()

