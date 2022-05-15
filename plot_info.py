# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 21:36:13 2022

@author: Julie S. Ingebrigtse
"""
import numpy as np
import matplotlib.pyplot as plt 

files = np.loadtxt('scint_mag_ori.txt', unpack=True)
scint = np.array(files[0, :])
mag5  = np.array(files[1, :])/12240
mag7  = np.array(files[2, :])/163200
ori5  = np.array(files[3, :])
ori7  = np.array(files[4, :])


# high_scint = np.loadtxt('30112019_highest_scint.txt', unpack=True)
# scint = np.array(high_scint[3, :])
# mag5  = np.array(high_scint[4, :])/12240
# mag7  = np.array(high_scint[5, :])/163200
# ori5  = np.array(high_scint[6, :])
# ori7  = np.array(high_scint[7, :])



# high_mag = np.loadtxt('08022019_highest_mag.txt', unpack=True)
# scint = np.array(high_mag[2, :])
# mag5  = np.array(high_mag[3, :])/12240
# mag7  = np.array(high_mag[4, :])/163200
# ori5  = np.array(high_mag[5, :])
# ori7  = np.array(high_mag[6, :])

print(max(mag5))


def linear_regression(x, y):     
    N = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    
    B1_num = ((x - x_mean) * (y - y_mean)).sum() #the slope
    B1_den = ((x - x_mean)**2).sum()
    B1 = B1_num / B1_den
    
    B0 = y_mean - (B1*x_mean) #the intercept 
    
    reg_line = 'y = {} + {}β'.format(B0, round(B1, 5))
    
    return (B0, B1, reg_line)


def corr_coef(x, y):
    ### return the correlation coefficient
    N = len(x)
    
    num = (N * (x*y).sum()) - (x.sum() * y.sum())
    den = np.sqrt((N * (x**2).sum() - x.sum()**2) * (N * (y**2).sum() - y.sum()**2))
    R = num / den
    return R




B0_5, B1_5, reg_line_mag5 = linear_regression(mag5, scint)
print('Regression Line 5: ', reg_line_mag5)
print('B1_5', B1_5)
R = corr_coef(mag5, scint)
print('Correlation Coef. k5: ', R)
print('"Goodness of Fit" k5: ', R**2)

B0_7, B1_7, reg_line_mag7 = linear_regression(mag7, scint)
print('Regression Line 7: ', reg_line_mag7)
print('B1_7', B1_7)
R = corr_coef(mag7, scint)
print('Correlation Coef. k7: ', R)
print('"Goodness of Fit": k7 ', R**2)






"""
Gradient VS scintillation for kernel 5
"""
text = '''Mean gradient: {} 
Mean scintillation: {}
y = {}x + {}'''.format(round(mag5.mean(), 4), 
                        round(scint.mean(), 3), 
                        round(B1_5, 5),
                        round(B0_5, 5))
plt.text(x=0.025, y=1.115, s=text, fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.2, 'pad': 9})
plt.scatter(mag5, scint, marker='o', alpha=0.3, color='violet')
plt.plot(mag5, B0_5 + B1_5*mag5, c = 'r', linewidth=1, alpha=.5, solid_capstyle='round') 
plt.suptitle('Gradient VS scintillation values, kernel 5', fontsize=14)
plt.title('08.02.2019 ', fontsize=13)
plt.xlabel('Normalized gradient magnitude values', fontsize=14)
plt.ylabel(r'Scintillation, 60 sec $\sigma$', fontsize=14)
# plt.xlim(right=0.0216, left=-0.001)
# plt.ylim(top=.4, bottom=0.19)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()





"""
Gradient VS scintillation for kernel 7
"""
# text1 = '''Mean gradient: {} 
# Mean scintillation: {}
# y = {}x + {}'''.format(round(mag7.mean(), 4), 
#                         round(scint.mean(), 3), 
#                         round(B1_7, 5),
#                         round(B0_7, 5))
# plt.text(x=0.0275, y=1.115, s=text1, fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.2, 'pad': 9})
# plt.scatter(mag7, scint, marker='o', alpha=0.3,color='dodgerblue')
# plt.plot(mag7, B0_7 + B1_7*mag7, c = 'b', linewidth=1, alpha=.5, solid_capstyle='round') 
# plt.suptitle('Gradient VS scintillation values, kernel 7', fontsize=14)
# plt.title('08.02.2019 ', fontsize=13)
# plt.xlabel('Normalized gradient magnitude values', fontsize=14)
# plt.ylabel(r'Scintillation, 60 sec $\sigma$', fontsize=14)
# # plt.xlim(left = -0.000092, right = 0.0025)
# # plt.ylim(top=0.4, bottom=0.19)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.show()



"""
Gradient VS scintillation for kernel 5 and 7
"""
# plt.scatter(mag7, scint, marker='o', alpha=0.3, color='dodgerblue', label='kernel 7')
# plt.plot(mag7, B0_7 + B1_7*mag7, c = 'b', linewidth=1, alpha=.5, solid_capstyle='round', label='reg kernel 7')
# plt.scatter(mag5, scint, marker='o', alpha=0.3, color='violet', label='kernel 5')
# plt.plot(mag5, B0_5 + B1_5*mag5, c = 'r', linewidth=1, alpha=.5, solid_capstyle='round', label='reg kernel 5')

# plt.suptitle('Gradient VS scintillation values', fontsize=14)
# plt.title('08.02.2019 ', fontsize=13)
# plt.xlabel('Normalized gradient magnitude values', fontsize=14)
# plt.ylabel(r'Scintillation, 60 sec $\sigma$', fontsize=14)
# # plt.xlim(left=-0.000092 ,right = 0.0025)
# # plt.ylim(top=0.4, bottom=0.19)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(loc=1, fontsize=14)
# plt.show()


###############################################################################


"""
Histogram for kernel 5 gradients 
"""
# plt.hist(mag5, density=True, bins=50, rwidth=0.7, color='violet', label='kernel 5')
# plt.title('Histogram of gradient values, kernel 5', fontsize=14)
# plt.xlabel('Normalized gradient values', fontsize=14)
# plt.yscale('log')
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=14)
# plt.show()





"""
Histogram for kernel 7 gradients 
"""
# plt.hist(mag7, density=True, bins=50, rwidth=0.7, color='dodgerblue', label='kernel 7')
# plt.title('Histogram of gradient values, kernel 7',fontsize=14)
# plt.xlabel('Normalized gradient values', fontsize=14)
# plt.yscale('log')
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=14)
# plt.show()





"""
Histogram for kernel 5 and 7 gradients 
"""
# plt.hist([mag5, mag7], density=True, rwidth=0.7, bins=50, color=['violet','dodgerblue'], label=['kernel 5', 'kernel 7'])
# plt.xlabel('Normalized gradient values', fontsize=14)
# plt.title('Histogram of gradient values', fontsize=14)
# plt.yscale('log')
# plt.legend(fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.show()


"""
Histogram for scintillations 
"""
# plt.hist(scint, density=True, bins=50, rwidth=0.7, color='coral', label='scint')
# plt.title('Histogram of scintillation values', fontsize=14)
# plt.xlabel('Scintillation values', fontsize=14)
# # plt.yscale('log')
# plt.legend(fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.show()

###############################################################################



# plt.scatter(liste, kernel_5, alpha=0.3, color='violet', label='kernel 5')
# plt.title('Scatter plot of gradient values')
# plt.legend()
# plt.show()





# plt.scatter(liste, kernel_7, alpha=0.3, color='dodgerblue', label='kernel 7')
# plt.title('Scatter plot of gradient values')
# plt.legend()
# plt.show()





# plt.scatter(liste, kernel_5, alpha=0.3,  color='violet', label='kernel 5')
# plt.scatter(liste, kernel_7, alpha=0.3,color='dodgerblue', label='kernel 7')
# plt.title('Scatter plot of gradient values')
# plt.legend()
# plt.show()