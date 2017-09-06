import numpy as np
import matplotlib.pyplot as plt

n = 20

# definition de a
a = np.zeros(n)
a[1] = 1

# visualisation de a
# on ajoute a droite la valeur de gauche pour la periodicite
plt.subplot(611)
plt.plot( np.append(a, a[0]) )

# calcul de A
A = np.fft.fft(a)
print(A,A[0])

# visualisation de A
# on ajoute a droite la valeur de gauche pour la periodicite
B = np.append(A, A[0])
plt.subplot(612)
plt.plot(np.real(B))
plt.plot(np.real(A))
plt.ylabel("partie reelle")

plt.subplot(613)
plt.plot(np.imag(B))
plt.plot(np.imag(A))
plt.ylabel("partie imaginaire")
plt.subplot(614)
plt.plot(np.real(A))
plt.ylabel("partie reelle")

plt.subplot(615)
plt.plot(np.imag(A))
plt.ylabel("partie imaginaire")

plt.show()