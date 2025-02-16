import math
import numpy as np
import matplotlib.pyplot as plt

def Mu(M):
    mu = np.arcsin(1 / M) * (180 / np.pi)
    return mu

def PrandtlMeyer(M, gamma):
    v = np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1))) - np.arctan(np.sqrt(M**2 - 1))
    return v

def moc2d(theta_max, theta_0, n):
    dtheta = (theta_max - theta_0) / (n - 1)
    node = 0.5 * n * (4 + n - 1)
    theta = np.zeros(int(node))
    v = np.zeros(int(node))
    KL = np.zeros(int(node))
    KR = np.zeros(int(node))
    
    for i in range(n):
        theta[i] = theta_0 + i * dtheta
        v[i] = theta[i]
        KL[i] = theta[i] - v[i]
        KR[i] = theta[i] + v[i]
    
    i = n
    theta[i] = theta[i - 1]
    v[i] = v[i - 1]
    KL[i] = KL[i - 1]
    KR[i] = KR[i - 1]
    
    p = 1
    q = n + 1
    for k in range(n - 1):
        j = p
        h = q
        theta[h] = 0
        KR[h] = KR[j]
        v[h] = KR[j] - theta[h]
        KL[h] = theta[h] - v[h]
        j += 1
        
        for i in range(h+1, n - p + q):
            KR[i] = KR[j]
            KL[i] = KL[i - 1]
            theta[i] = 0.5 * (KL[i] + KR[i])
            v[i] = 0.5 * (KR[i] - KL[i])
            j += 1
        
        if i == n - p + q-1:
            h = i + 1
        else:
            h += 1
        
        theta[h] = theta[h - 1]
        v[h] = v[h - 1]
        KL[h] = KL[h - 1]
        KR[h] = KR[h - 1]
        
        p += 1
        q = h + 1
    
    return v, KL, KR, theta

def get_mach_number(nu, gp1, gm1):

    convdr = math.pi / 180.0  # Conversion factor from degrees to radians
    nur = nu * convdr         # Convert Prandtl-Meyer angle to radians

    msm1o = 1.0               # Initial guess for Mach Number
    nuo = math.sqrt(gp1 / gm1) * math.atan(math.sqrt(gm1 * msm1o / gp1)) - math.atan(math.sqrt(msm1o))
    msm1n = msm1o + 0.01      # iterate

    # Iterate until the difference is within the tolerance
    while abs(nur - nuo) > 0.001:
        nun = math.sqrt(gp1 / gm1) * math.atan(math.sqrt(gm1 * msm1n / gp1)) - math.atan(math.sqrt(msm1n))
        deriv = (nun - nuo) / (msm1n - msm1o)
        nuo = nun
        msm1o = msm1n
        msm1n = msm1o + (nur - nuo) / deriv

    mach1 = math.sqrt(msm1o + 1.0)  # Final Mach number
    return mach1


# Gas properties
gamma = float(input('Enter gamma : '))
gp1 = gamma + 1
gm1 = gamma - 1

# Design parameters
Me = float(input('Enter exit Mach no. : '))
theta_max = PrandtlMeyer(Me, gamma) * 180 / (2 * np.pi)

# Incident expansion wave conditions
n = int(input('Enter number of characteristics lines (greater than 2) emanating from sharp corner throat : '))
theta_0 = theta_max / n

# Characteristic parameter solver
v, KL, KR, theta = moc2d(theta_max, theta_0, n)

# Mach number and Mach angle at each node
node = int(0.5 * n * (4 + n - 1))
M = np.zeros(node)
mu = np.zeros(node)
for i in range(node):
    M[i] = get_mach_number(v[i],gp1,gm1)
    print(M[i])
    mu[i] = Mu(M[i])

# Grid generator
plt.figure(1)
D = 1  # Non-Dimensional y coordinate of throat wall
i = 0
x = np.zeros(node)
y = np.zeros(node)
wall = theta_max
while i <= n:
    if i == 0:
        x[i] = -D / np.tan(np.radians(theta[i] - mu[i]))
        y[i] = 0
        plt.plot([0, x[i]], [D, 0])
    elif i == n:
        x[i] = (y[i - 1] - D - x[i - 1] * np.tan(np.radians((theta[i - 1] + theta[i] + mu[i - 1] + mu[i]) * 0.5))) / (np.tan(np.radians(0.5 * (wall + theta[i]))) - np.tan(np.radians((theta[i - 1] + theta[i] + mu[i - 1] + mu[i]) * 0.5)))
        y[i] = D + x[i] * np.tan(np.radians(0.5 * (wall + theta[i])))
        plt.plot([x[i - 1], x[i]], [y[i - 1], y[i]])
        plt.plot([0, x[i]], [D, y[i]])
    else:
        x[i] = (D - y[i - 1] + x[i - 1] * np.tan(np.radians(0.5 * (mu[i - 1] + theta[i - 1] + mu[i] + theta[i])))) / (np.tan(np.radians(0.5 * (mu[i - 1] + theta[i - 1] + mu[i] + theta[i]))) - np.tan(np.radians(theta[i] - mu[i])))
        y[i] = np.tan(np.radians(theta[i] - mu[i])) * x[i] + D
        plt.plot([x[i - 1], x[i]], [y[i - 1], y[i]])
        plt.plot([0, x[i]], [D, y[i]])
    i += 1

h = i
k = 0
i = h
for j in range(1, n - 1):
    while i <= h + n - k - 1:
        if i == h:
            x[i] = x[i - n + k] - y[i - n + k] / np.tan(np.radians(0.5 * (theta[i - n + k] + theta[i] - mu[i - n + k] - mu[i])))
            y[i] = 0
            plt.plot([x[i - n + k], x[i]], [y[i - n + k], y[i]])
        elif i == h + n - k - 1:
            x[i] = (x[i - n + k] * np.tan(np.radians(0.5 * (theta[i - n + k] + theta[i]))) - y[i - n + k] + y[i - 1] - x[i - 1] * np.tan(np.radians((theta[i - 1] + theta[i] + mu[i - 1] + mu[i]) * 0.5))) / (np.tan(np.radians(0.5 * (theta[i - n + k] + theta[i]))) - np.tan(np.radians((theta[i - 1] + theta[i] + mu[i - 1] + mu[i]) * 0.5)))
            y[i] = y[i - n + k] + (x[i] - x[i - n + k]) * np.tan(np.radians(0.5 * (theta[i - n + k] + theta[i])))
            plt.plot([x[i - 1], x[i]], [y[i - 1], y[i]])
            plt.plot([x[i - n + k], x[i]], [y[i - n + k], y[i]])
        else:
            s1 = np.tan(np.radians(0.5 * (theta[i] + theta[i - 1] + mu[i] + mu[i - 1])))
            s2 = np.tan(np.radians(0.5 * (theta[i] + theta[i - n + k] - mu[i] - mu[i - n + k])))
            x[i] = (y[i - n + k] - y[i - 1] + s1 * x[i - 1] - s2 * x[i - n + k]) / (s1 - s2)
            y[i] = y[i - 1] + (x[i] - x[i - 1]) * s1
            plt.plot([x[i - 1], x[i]], [y[i - 1], y[i]])
            plt.plot([x[i - n + k], x[i]], [y[i - n + k], y[i]])
        i += 1
    k += 1
    h = i
    i = h

plt.title(f'Characteristic lines for Mach={Me} and gamma={gamma}')
plt.xlabel('x/x0')
plt.ylabel('y/y0')
plt.axis('equal')
plt.xlim([0, x[node - 1] + 0.5])
plt.ylim([0, y[node - 1] + 0.5])

# Nozzle coordinates (x_wall, y_wall)
x_wall = np.zeros(n)
y_wall = np.zeros(n)
x_wall[0] = 0
y_wall[0] = 1
i = 1
j = n
while i < n:
    x_wall[i] = x[j]
    y_wall[i] = y[j]
    print("X=",x_wall[i],"Y=",y_wall[i])
    j += (n - i + 1)
    i += 1

plt.figure(2)
plt.plot(x_wall, y_wall)
print(x_wall)
plt.show()

