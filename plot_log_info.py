# -*- coding: utf-8 -*-
"""
Created on Sat May 14 19:54:07 2022

@author: Julie S. Ingebrigtse
"""
import numpy as np
import matplotlib.pyplot as plt 

"This file is for all the dates"
files = np.loadtxt('scint_mag_ori.txt', unpack=True)
files0 = np.loadtxt('scint_mag_ori_nullnull.txt', unpack=True)
scint = np.array(files0[0, :])
scint = np.log(scint)

mag5  = np.array(files0[1, :])/12240
mag5 = np.log(mag5)
    
mag7  = np.array(files0[2, :])/163200
mag7  = np.log(mag7)


def corr_coef(x, y):
    ### return the correlation coefficient
    N = len(x)
    
    num = (N * (x*y).sum()) - (x.sum() * y.sum())
    den = np.sqrt((N * (x**2).sum() - x.sum()**2) * (N * (y**2).sum() - y.sum()**2))
    R = num / den
    return R


R = corr_coef(mag5, scint)
print('Correlation Coef. k5: ', R)
print('"Goodness of Fit" k5: ', R**2)

R = corr_coef(mag7, scint)
print('Correlation Coef. k7: ', R)
print('"Goodness of Fit": k7 ', R**2)

# a, b = np.polyfit(mag5, scint, 1)
# text = '''y = {}x + {}'''.format(round(a, 5), round(b, 5))
# plt.text(x=-9.1, y=0.8, s=text, fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.2, 'pad': 9})
# plt.plot(mag5, a*mag5 + b, c='r')
# plt.scatter(mag5, scint,  marker='o', alpha=0.3, color='violet')
# plt.suptitle('Gradient VS scintillation values, kernel 5', fontsize=14)
# plt.title('Double logaritmic',fontsize=13)
# plt.xlabel('Normalized gradient magnitude values', fontsize=14)
# plt.ylabel(r'Scintillation, 60 sec $\sigma$', fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.show()



# a, b = np.polyfit(mag7, scint, 1)
# text = '''y = {}x + {}'''.format(round(a, 5), round(b, 5))
# plt.text(x=-11.3, y=0.8, s=text, fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.2, 'pad': 9})
# plt.plot(mag7, a*mag7 + b, c='b')
# plt.scatter(mag7, scint,  marker='o', alpha=0.3, color='dodgerblue')
# plt.suptitle('Gradient VS scintillation values, kernel 7', fontsize=14)
# plt.title('Double logaritmic',fontsize=13)
# plt.xlabel('Normalized gradient magnitude values', fontsize=14)
# plt.ylabel(r'Scintillation, 60 sec $\sigma$', fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.show()



a5, b5 = np.polyfit(mag7, scint, 1)
a7, b7 = np.polyfit(mag7, scint, 1)
plt.plot(mag7, a5*mag7 + b7, c='b', label='reg kernel 7')
plt.plot(mag5, a5*mag5 + b5, c='r', label='reg kernel 5')


plt.scatter(mag7, scint,  marker='o', alpha=0.3, color='dodgerblue', label='kernel 7')
plt.scatter(mag5, scint,  marker='o', alpha=0.3, color='violet', label='kernel 5')
plt.suptitle('Gradient VS scintillation values', fontsize=14)
plt.title('Double logaritmic',fontsize=13)
plt.xlabel('Normalized gradient magnitude values', fontsize=14)
plt.ylabel(r'Scintillation, 60 sec $\sigma$', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.xlim(left=-8,right = -6)
# plt.ylim(top=-1.2, bottom=-1.4)
plt.legend(loc='best', fontsize=14)
plt.show()
