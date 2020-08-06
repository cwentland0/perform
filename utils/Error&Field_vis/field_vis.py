# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 18:36:48 2020

@author: ashis
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio

def calc_RAE(truth,pred):
    
    RAE = np.mean(np.abs(truth-pred))/np.max(np.abs(truth))
    
    return RAE

pod = True #change to True to plot
dec = False
enc = False
labs = ['solforCons','solforPrim','Dec Jac','Enc Jac']

#Field-plots
field_plots = True #change for switch on/off
fp_ts = [1000,2000,3000,4000] #time-instances to plot 

#Field movies
field_movie = True 

st = 0      #first/last frame
end = 10000
inter = 100 #frame interval

#Error plots/movies
error_plots = True
error_movie = False

nx = 256
dt = 1e-8
x = np.linspace(0,0.01,num=nx)
file_path = './Datasets/'
file_path_figs = './Figure/'

#loading the solutions

sol_FOM = np.load(file_path+'solPrim_FOM.npy')


if(pod):
    sol_POD = np.load(file_path+'solPrim_POD.npy')
    
if(dec):
    sol_dec = np.load(file_path+'solPrim_dec.npy')
    
if(enc):
    sol_enc = np.load(file_path+'solPrim_enc.npy')
    
    
if(field_plots):
    
    for i in fp_ts:
        
        f, axs = plt.subplots(2)
        
        axs[0].plot(x,sol_FOM[:,0,i],'-k',label=labs[0])
        
        if(pod):
            axs[0].plot(x,sol_POD[:,0,i],'-r',label=labs[1])
            
        if(dec):
            axs[0].plot(x,sol_dec[:,0,i],'-b',label=labs[2])

        if(enc):
            axs[0].plot(x,sol_enc[:,0,i],'-g',label=labs[3])

        axs[0].set_xlabel('x')
        axs[0].set_ylabel('P (Pa)')
        axs[0].set_ylim([np.amin(sol_FOM[:,0,st:end]),np.amax(sol_FOM[:,0,st:end])])
        axs[0].set_xlim([0, 0.01])
        axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        axs[0].legend(loc='upper left')

    
        axs[1].plot(x,sol_FOM[:,1,i],'-k',label=labs[0])
        
        if(pod):
            axs[1].plot(x,sol_POD[:,1,i],'-r',label=labs[1])
            
        if(dec):
            axs[1].plot(x,sol_dec[:,1,i],'-b',label=labs[2])

        if(enc):
            axs[1].plot(x,sol_enc[:,1,i],'-g',label=labs[3])

        axs[1].set_xlabel('x')
        axs[1].set_ylabel('U (m/s)')
        axs[1].set_ylim([np.amin(sol_FOM[:,1,st:end]),np.amax(sol_FOM[:,1,st:end])])
        axs[1].set_xlim([0, 0.01])
        axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    
    
        for ax in f.get_axes():
            ax.label_outer()
        
        plt.show()
        f.savefig(file_path_figs+'Field Plots/P1_P2_'+str(i)+'.png')
        plt.close()
    
    
        f, axs = plt.subplots(2)
        
        axs[0].plot(x,sol_FOM[:,2,i],'-k',label=labs[0])
        
        if(pod):
            axs[0].plot(x,sol_POD[:,2,i],'-r',label=labs[1])
            
        if(dec):
            axs[0].plot(x,sol_dec[:,2,i],'-b',label=labs[2])

        if(enc):
            axs[0].plot(x,sol_enc[:,2,i],'-g',label=labs[3])

        axs[0].set_xlabel('x')
        axs[0].set_ylabel('T (K)')
        axs[0].set_ylim([np.amin(sol_FOM[:,2,st:end]),np.amax(sol_FOM[:,2,st:end])])
        axs[0].set_xlim([0, 0.01])
        axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        axs[0].legend(loc='upper left')

    
        axs[1].plot(x,sol_FOM[:,3,i],'-k',label=labs[0])
        
        if(pod):
            axs[1].plot(x,sol_POD[:,3,i],'-r',label=labs[1])
            
        if(dec):
            axs[1].plot(x,sol_dec[:,3,i],'-b',label=labs[2])

        if(enc):
            axs[1].plot(x,sol_enc[:,3,i],'-g',label=labs[3])

        axs[1].set_xlabel('x')
        axs[1].set_ylabel('U (m/s)')
        axs[1].set_ylim([np.amin(sol_FOM[:,3,st:end]),np.amax(sol_FOM[:,3,st:end])])
        axs[1].set_xlim([0, 0.01])
        axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    
    
        for ax in f.get_axes():
            ax.label_outer()
        
        plt.show()
        f.savefig(file_path_figs+'Field Plots/P3_P4_'+str(i)+'.png')
        plt.close()

    
if(field_movie):
    
    images=[]
    images1=[]
    
    for i in range(st,end):
    
        if((i % inter) != 0):
            continue


        f, axs = plt.subplots(2)
        
        axs[0].plot(x,sol_FOM[:,0,i],'-k',label=labs[0])
        
        if(pod):
            axs[0].plot(x,sol_POD[:,0,i],'-r',label=labs[1])
            
        if(dec):
            axs[0].plot(x,sol_dec[:,0,i],'-b',label=labs[2])

        if(enc):
            axs[0].plot(x,sol_enc[:,0,i],'-g',label=labs[3])

        axs[0].set_xlabel('x')
        axs[0].set_ylabel('P (Pa)')
        axs[0].set_ylim([np.amin(sol_FOM[:,0,st:end]),np.amax(sol_FOM[:,0,st:end])])
        axs[0].set_xlim([0, 0.01])
        axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        axs[0].legend(loc='upper left')

    
        axs[1].plot(x,sol_FOM[:,1,i],'-k',label=labs[0])
        
        if(pod):
            axs[1].plot(x,sol_POD[:,1,i],'-r',label=labs[1])
            
        if(dec):
            axs[1].plot(x,sol_dec[:,1,i],'-b',label=labs[2])

        if(enc):
            axs[1].plot(x,sol_enc[:,1,i],'-g',label=labs[3])

        axs[1].set_xlabel('x')
        axs[1].set_ylabel('U (m/s)')
        axs[1].set_ylim([np.amin(sol_FOM[:,1,st:end]),np.amax(sol_FOM[:,1,st:end])])
        axs[1].set_xlim([0, 0.01])
        axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    
    
        for ax in f.get_axes():
            ax.label_outer()

        
        f.canvas.draw()
        image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')
        print(i)
        images.append(image.reshape(f.canvas.get_width_height()[::-1] + (3,)))
        plt.close()
    

        f, axs = plt.subplots(2)
        
        axs[0].plot(x,sol_FOM[:,2,i],'-k',label=labs[0])
        
        if(pod):
            axs[0].plot(x,sol_POD[:,2,i],'-r',label=labs[1])
            
        if(dec):
            axs[0].plot(x,sol_dec[:,2,i],'-b',label=labs[2])

        if(enc):
            axs[0].plot(x,sol_enc[:,2,i],'-g',label=labs[3])

        axs[0].set_xlabel('x')
        axs[0].set_ylabel('T (K)')
        axs[0].set_ylim([np.amin(sol_FOM[:,2,st:end]),np.amax(sol_FOM[:,2,st:end])])
        axs[0].set_xlim([0, 0.01])
        axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        axs[0].legend(loc='upper left')

    
        axs[1].plot(x,sol_FOM[:,3,i],'-k',label=labs[0])
        
        if(pod):
            axs[1].plot(x,sol_POD[:,3,i],'-r',label=labs[1])
            
        if(dec):
            axs[1].plot(x,sol_dec[:,3,i],'-b',label=labs[2])

        if(enc):
            axs[1].plot(x,sol_enc[:,3,i],'-g',label=labs[3])

        axs[1].set_xlabel('x')
        axs[1].set_ylabel('U (m/s)')
        axs[1].set_ylim([np.amin(sol_FOM[:,3,st:end]),np.amax(sol_FOM[:,3,st:end])])
        axs[1].set_xlim([0, 0.01])
        axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    
    
        for ax in f.get_axes():
            ax.label_outer()
        
        f.canvas.draw()
        image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')
        images1.append(image.reshape(f.canvas.get_width_height()[::-1] + (3,)))
    
        plt.close()
    
    kwargs_write = {'fps':10.0, 'quantizer':'nq'}
    imageio.mimsave(file_path_figs+'Field Movies/'+'P_1_2.gif', images, fps=100)
    
    kwargs_write = {'fps':1.0, 'quantizer':'nq'}
    imageio.mimsave(file_path_figs+'Field Movies/'+'P_3_4.gif', images1, fps=100)


if(error_plots or error_movie):
    
    #calculate RAE
    if(dec):
        RAE_dec = np.zeros((end-st,4))
    if(pod):
        RAE_pod = np.zeros((end-st,4))
    if(enc):
        RAE_enc = np.zeros((end-st,4))
    
    for i in range(st,end):
        for j in range(4):
            
            if(dec):
                RAE_dec[i-st,j] = calc_RAE(sol_dec[:,j,i],sol_FOM[:,j,i])
            if(enc):
                RAE_enc[i-st,j] = calc_RAE(sol_enc[:,j,i],sol_FOM[:,j,i])
            if(pod):
                RAE_pod[i-st,j] = calc_RAE(sol_POD[:,j,i],sol_FOM[:,j,i])
                

if(error_plots):
    
    t = np.linspace(st,end,end-st)*(dt/1e-6)
    labels = ['Pressure','Velocity','Temperature','Y1']
    for i in range(4):
        f,axs = plt.subplots(1)
        f.suptitle(labels[i])
        if(dec):
            axs.plot(t,RAE_dec[:,i]*100,'--b',label=labs[2])
        if(enc):
            axs.plot(t,RAE_enc[:,i]*100,'--g',label=labs[3])
        if(pod):
            axs.plot(t,RAE_pod[:,i]*100,'--r',label=labs[1])
        
        
        axs.legend()
        axs.set_xlabel('Time ($\mu$s)')
        axs.set_ylabel('RAE(%)')
        plt.show()
        f.savefig(file_path_figs+'Error Plots/'+labels[i]+'.png')


if(error_movie):
    
    labels = ['Pressure','Velocity','Temperature','Y1']
    
    for prim_num in range(4):
        
        images=[]
        for i in range(st,end):
            
            if((i % inter) != 0):
                continue
            
            print(i)
            f,axs = plt.subplots(1)
            f.suptitle(labels[prim_num])
            if(dec):
                axs.plot(t,RAE_dec[:,prim_num]*100,'--b',label=labs[2])
            if(enc):
                axs.plot(t,RAE_enc[:,prim_num]*100,'--g',label=labs[3])
            if(pod):
                axs.plot(t,RAE_pod[:,prim_num]*100,'--r',label=labs[1])
        
            axs.axvline(x=250,linestyle='--')
            axs.axvline(x=i,linestyle='-')
            
            
            axs.legend(loc=1)
            axs.set_xlabel('Time($\mu$s)')
            axs.set_ylabel('RAE(%)')
            axs.set_frame_on(False)
            axs.set_frame_on(True)
            f.canvas.draw()
            image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')
            images.append(image.reshape(f.canvas.get_width_height()[::-1] + (3,)))
           
        kwargs_write = {'fps':1.0, 'quantizer':'nq'}
        imageio.mimsave(file_path_figs+'Error Plots/'+labels[prim_num]+'_err_mov.gif', images, fps=100)    


            
            
            

    
    

    
