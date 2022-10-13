#Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


'''Define a function with the equation of a gaussian curve (2.4)'''

def gaussian(x, I, X, w):
    return (I/(w*np.sqrt(2*np.pi)))*np.exp(-((x-X)**2)/(2*w**2))

'''Define a function for working out parametres for the curve fit'''

def Params(x,y,xmin,xmax):
    curvepoints = ((x>xmin)*(x<xmax))
    ybound=y[curvepoints]
    xbound=x[curvepoints]
    X=np.mean(xbound)
    maxy = max(ybound)
    w = (max(xbound)-min(xbound))
    xfit = np.linspace(xmin,xmax,401)
    popt, pcov = curve_fit(gaussian, xbound, ybound, p0=[maxy,X,w])
    perr = np.sqrt(np.diag(pcov))
    return gaussian(xfit, *popt),popt[1],popt[2],popt[0],perr

'''Define equation for background data (2.5)'''

def backgroundcurve(x,I1,x0):
    return I1 - (((x-x0)**2)/100)

'''Finding the FWHM using widths from gaussian curve fit'''

def FullWHM(widths):
    fwhm = (2*np.sqrt(2*np.log(2)))*widths
    return fwhm

'''Finding the uncertainty in the FWHM using the errors in Widths'''

def FWHMErr(err0):
    FWHMErr_ = (2*np.sqrt(2*np.log(2)))*err0
    return FWHMErr_
    
'''Finding d for each plane in Angstroms using Braggs Law'''

def finding_d(plane,wavelength):
    Theta_rad = (plane*(np.pi/180))/2
    d = (wavelength) /( 2 * np.sin(Theta_rad))
    return d    

'''Finding error for d by taking the partial derivitive of braggs law with respect to theta
Then use the uncertainty equation'''

def D_error(err1,wavelength,err2):
    Theta_rad_err = (err1*(np.pi/180))/2
    Theta_rad = (err2*(np.pi/180))/2
    derr = np.sqrt((-0.5*(np.cos(Theta_rad)/((np.sin(Theta_rad))**2))*Theta_rad_err)**2)
    return derr

'''Find approx values for hkl '''

def hkl_approx(d_peak,a_estimate):
    hkl_estimate = (a_estimate**2)/(d_peak**2)
    return hkl_estimate
    
'''Using final sum of squares of hkl, find a precise average value of a'''

def precise_a(d_peak,sos):
    a = np.sqrt(sos*(d_peak**2))
    return a

            
'''Determine compotition using eq (2.3)'''

def comp(a,aAu,aCu):
    x = (a-aCu)/(aAu-aCu)
    return x

'''Estimate the crystalite size using eq (2.4)'''

def grain(wavelength,fwhm,angle):
    theta_rad = np.cos(angle*(np.pi/180)/2)
    fwhm_rad = fwhm*(np.pi/180)
    grain_size = wavelength/(fwhm_rad*theta_rad)
    return grain_size


'''Calculate the error in the grain size by partial differentiation with respect
to Theta and the FWHM'''

def grainerr(wavelength,fwhm,angle,fwhM,two_theta):
    theta_rad = (angle*(np.pi/180)/2)
    fwhm_rad = fwhm*(np.pi/180)
    theta_rad_err = (two_theta*(np.pi/180)/2)
    fwhm_rad_err = fwhM*(np.pi/180)
    term1 = ((wavelength/np.cos(theta_rad))**2)*fwhm_rad_err
    term2 = ((wavelength*np.sin(theta_rad))/(fwhm_rad*(np.cos(theta_rad))**2))*theta_rad_err
    grain_size_err = np.sqrt(term1 + term2)
    return grain_size_err




def ProcessData(filename):

    '''Values of the lattice parameters of gold and copper'''
    
    aAu = 0.40782 
    aCu = 0.36149

    '''Open the file with exception handling
    Obtain the wavelength value in nm and the number of lines of metadata'''
    
    try:
        with open(filename,'r') as data:
            for line_number,line in enumerate(data):
                line = line.strip()
                if line.startswith('&END'):
                    lines_count = line_number
                if line.startswith('Wavelength'):
                    wavelength = 0.1*float(line.strip('Wavelength (Angstroms)='))
    except IOError:
        print('An IO Error has occured')
    except FileNotFoundError:
        print('File could not be found')
    
    '''Importing data and seperating it into seperate arrays using the number of lines
    of metadata found in the try block'''
    
    data = np.genfromtxt(filename, skip_header = (lines_count + 2))
    
    x = data[:,0]
    y = data[:,1]
        
    '''Plot the raw data and customize the graph'''

    plt.plot(x, y, 'b.', label='Data')
    plt.xlabel('Diffraction Angle 2\u03B8(\u00b0)')
    plt.ylabel('Counts')
    plt.ylim(1,(max(y)+1000))
    plt.yscale('log')
    plt.title('py18er', loc = 'center')
    
    '''Find the the peaks of the data points and show this on the graph'''
    
    peaks,_ = find_peaks(y, height=100, width = 2)
    plt.plot(x[peaks], y[peaks] , 'x')
       
    '''Create an array of x coordinates of the peaks with seperate arrays for 
    xmin and xmax values (points around the curve)'''
    
    xpeaks = np.array(x[peaks])
    highx = xpeaks + 2
    lowx = xpeaks - 2
    
    '''Combine the xmin and xmax values into one list'''
    
    lst = []
    for xmax in highx:
        for xmin in lowx:  
           if xmax == (xmin+4):
               lst.append((xmin, xmax))
    
    '''Define lists for data to be extracted too from the fitting functions
    popt and pcov values'''
    
    Widths = []
    Two_Theta = []
    Area_under_peak = []
    Errors = []     
    
    '''Use the xmin and xmax values in the fitting function to fit lines to all the peaks
    Append data to lists and plot the curve fits'''
    
    for xmin, xmax in lst:
        yfit = Params(x,y,xmin,xmax)[0]
        Two_Theta.append((Params(x,y,xmin,xmax)[1]))
        Widths.append((Params(x,y,xmin,xmax)[2]))
        Area_under_peak.append((Params(x,y,xmin,xmax)[3]))
        Errors.append(Params(x,y,xmin,xmax)[4])   
        xfit = np.linspace(xmin,xmax,401)
        plt.plot(xfit,yfit,'m-')
    
    '''Define seperate lists for errors in each parameter'''
    
    Err_Area = []
    Err_Two_Theta = []
    Err_Width = []
    
    '''Append the correct values to each list of errors'''
    
    for i in Errors:
        Err_Area.append(i[0])
        Err_Two_Theta.append(i[1])
        Err_Width.append(i[2])
            
    '''create a list and for every x value above the xmax and below the xmin,
    append that to a boolean array 
    bgtf is backgroundtruefalse'''
    
    bgtf = []
    for i in range(len(xpeaks)):
         leftside = x<(lowx[i])
         rightside = x>(highx[i])
         bgtf.append(leftside | rightside)
    
    '''combine all the individual boolean arrays into one by iterating over a while loop.
    Now all the points around each curve are marked as True'''
    
    i = 1
    backgroundindex = bgtf[0]
    while True:
        backgroundindex = np.multiply(backgroundindex,bgtf[i])
        i += 1
        if i is len(bgtf):
            break
        
    '''Index the background data for each axis'''

    backgroundy = y[backgroundindex]
    backgroundx = x[backgroundindex]
    
    '''Plot background curve'''
    
    popt, pcov = curve_fit(backgroundcurve, backgroundx, backgroundy, p0=[max(backgroundy),80])
    perr = np.sqrt(np.diag(pcov))
    for I0,X0,z in zip('IX',popt,perr):
        I1 = popt[0]
        x0 = popt[1]
    yfit1 = backgroundcurve(backgroundx, *popt)
    plt.plot(backgroundx,yfit1, 'g-',label='Background')
        
    '''Create background data'''
    
    Iback = I1 - (((x-x0)**2)/100)
    
    '''Subtract background from original data'''
    
    ycorrected = y - Iback
    
    '''Plot corrected data'''
    
    plt.plot(x,ycorrected,'r.',label='Corrected Signal')
    plt.legend()
            
    '''Calculate the FWHM and uncertainty in FWHM'''
    
    FWHM = [] 
    for widths in Widths:
        FWHM.append(FullWHM(widths))
        
    Err_FWHM = []
    for err0 in Err_Width:
        Err_FWHM.append(FWHMErr(err0))    
    
    '''Calculate lattice spacing d and uncertainty in d'''
    d = []
    for plane in Two_Theta:
        d.append(finding_d(plane,wavelength))
                
    Err_d = []
    for n1,err1 in enumerate(Err_Two_Theta):
        for n2,err2 in enumerate(Two_Theta):
            if n1 == n2:
                Err_d.append(D_error(err1,wavelength,err2))
        
    '''Finding an estimate for lattice parameter assuming equal composition'''
        
    a_estimate = (aAu + aCu)/2
    
    '''Create a list of all possible miller indices for a FCC.
     hkl must be all even or all odd'''
     
    all_hkl = np.array([(1,1,1),(2,0,0),(2,2,0),(2,2,2),(3,1,1),(3,3,1),(3,3,3),(4,0,0),(4,2,0),(4,4,0),(4,4,4),(4,2,2)])
    all_hkl = list(np.reshape(all_hkl,(12,3)))
    all_sum_of_squares = [3,4,8,12,11,19,27,16,20,32,48,24]
    
    '''Round all estimates down and append to hkl list of all actual sum of squares'''
    sum_of_squares_peaks = []
    for d_peak in d:
        sum_of_squares_peaks.append(int(hkl_approx(d_peak,a_estimate)))
        
    '''For each actual sum of squares value, find the corresponding incides by enumerating'''
    
    hkl_peaks_enumerated = []
    for num,el in enumerate(all_sum_of_squares):
        for i in sum_of_squares_peaks:
            if el == i:
                hkl_peaks_enumerated.append(num)
    
    final_hkl = []           
    for i in hkl_peaks_enumerated:
        final_hkl.append(all_hkl[i])
    
    '''Calculate a more precise value of a using the actual miller indices'''

    final_a = []
    for n1,d_peak in enumerate(d):
        for n2,sos in enumerate(sum_of_squares_peaks):
            if n1 == n2:
                final_a.append(precise_a(d_peak,sos))
                
    '''Find a final value for a and the uncertainty'''
    
    a = np.mean(final_a)
    
    a_err = np.std(final_a)

    '''Calculate the composition and its uncertainty'''
    
    composition = comp(a,aAu,aCu)
    
    comp_err = composition*(a_err/a)
    
    '''Calculate grain size'''
    
    grain_size = []
    for n1, fwhm in enumerate(FWHM):
        for n2, angle in enumerate(Two_Theta):
            if n1 == n2:
                grain_size.append(grain(wavelength,fwhm,angle))
            
    '''Calculate a final grain size'''
                
    final_grainsize = np.mean(grain_size)
    
    '''Calculate uncertainty in grain size by the average of the uncertainty for each one'''
    
    Err_grain = []
    for n1, fwhm in enumerate(Err_FWHM):
        for n2, angle in enumerate(Err_Two_Theta):
                for n3, fwhM in enumerate(FWHM):
                    for n4, two_theta in enumerate(Two_Theta):
                        if n1 == n2 == n3 == n4:
                            Err_grain.append(grainerr(wavelength,fwhm,angle,fwhM,two_theta))
    
    err_grain = np.mean(Err_grain)
    
    '''Formatting all the data ready to be plotted'''
    
    Two_Theta_plot = []
    for i in Two_Theta:
        Two_Theta_plot.append(round(i,3))
        
    FWHM_plot = []
    for i in FWHM:
        FWHM_plot.append(round(i,3))
        
    Area_plot = []
    for i in Area_under_peak:
        Area_plot.append(round(i,1))
    
    '''Annotate all peaks with data
    Rough annotations without uncertainties or ideal formatting'''
    
    for i in range(0,len(peaks)):
        plt.annotate(str(final_hkl[i]) + str(' at ') + str(Two_Theta_plot[i]) + str('\n')
        + str('FWHM ') + str(FWHM_plot[i]) + str('\n') + str('Area ') + str(Area_plot[i])
        ,(x[peaks][i],y[peaks][i]),
        xytext = (xpeaks[i]+3,y[peaks][i]+50)
        ,arrowprops = {'facecolor' : 'black'})
    
    '''Rounding other calculated parameters'''
    a_plot = round(a,3)
    comp_plot = round(composition,3)
    grain_plot = round(final_grainsize,3)
    
    '''Adding a text box showing rounded information about the lattice'''
    
    plt.text(10,(max(y)+250),'Lattice Parameter (nm) = ' + str(a_plot) + str('\n')
    + str('Composition = ') + str(comp_plot) + str('\n') + str('Grain Size (nm) = ')
    + str(grain_plot))



    results={"Peaks": [ 
        #This is a list of dictionaries for each peak
        {
            # Location of peak in two theta in dgrees, followed 
            # by the stadnard error
            "theta": Two_Theta[0], 
            "2theta_error":Err_Two_Theta[0],
            #Value of d-space for this peak and it's stadnard 
            # error (nm)
            "d": d[0], 
            "d_error": Err_d[0], 
            #Value of the FWHM for first peak & error (degrees)
            "FWHM": FWHM[0], 
            "FWHM_erroe":Err_FWHM[0],
            #Value of the area under the peak peak  & error
            "Area": Area_under_peak[0],
            "Area_error":Err_Area[0]
        },
                {
            # Location of peak in two theta in dgrees, followed 
            # by the stadnard error
            "theta": Two_Theta[1], 
            "2theta_error":Err_Two_Theta[1],
            #Value of d-space for this peak and it's stadnard 
            # error (nm)
            "d": d[1], 
            "d_error": Err_d[1], 
            #Value of the FWHM for first peak & error (degrees)
            "FWHM": FWHM[1], 
            "FWHM_erroe":Err_FWHM[1],
            #Value of the area under the peak peak  & error
            "Area": Area_under_peak[1],
            "Area_error":Err_Area[1]
        },
                                {
            # Location of peak in two theta in dgrees, followed 
            # by the stadnard error
            "theta": Two_Theta[2], 
            "2theta_error":Err_Two_Theta[2],
            #Value of d-space for this peak and it's stadnard 
            # error (nm)
            "d": d[2], 
            "d_error": Err_d[2], 
            #Value of the FWHM for first peak & error (degrees)
            "FWHM": FWHM[2], 
            "FWHM_erroe":Err_FWHM[2],
            #Value of the area under the peak peak  & error
            "Area": Area_under_peak[2],
            "Area_error":Err_Area[2]
        },
                                                {
            # Location of peak in two theta in dgrees, followed 
            # by the stadnard error
            "theta": Two_Theta[3], 
            "2theta_error":Err_Two_Theta[3],
            #Value of d-space for this peak and it's stadnard 
            # error (nm)
            "d": d[3], 
            "d_error": Err_d[3], 
            #Value of the FWHM for first peak & error (degrees)
            "FWHM": FWHM[3], 
            "FWHM_erroe":Err_FWHM[3],
            #Value of the area under the peak peak  & error
            "Area": Area_under_peak[3],
            "Area_error":Err_Area[3]
        },
                                                                {
            # Location of peak in two theta in dgrees, followed 
            # by the stadnard error
            "theta": Two_Theta[4], 
            "2theta_error":Err_Two_Theta[4],
            #Value of d-space for this peak and it's stadnard 
            # error (nm)
            "d": d[4], 
            "d_error": Err_d[4], 
            #Value of the FWHM for first peak & error (degrees)
            "FWHM": FWHM[4], 
            "FWHM_erroe":Err_FWHM[4],
            #Value of the area under the peak peak  & error
            "Area": Area_under_peak[4],
            "Area_error":Err_Area[4]
        },
                                                                                {
            # Location of peak in two theta in dgrees, followed 
            # by the stadnard error
            "theta": Two_Theta[5], 
            "2theta_error":Err_Two_Theta[5],
            #Value of d-space for this peak and it's stadnard 
            # error (nm)
            "d": d[5], 
            "d_error": Err_d[5], 
            #Value of the FWHM for first peak & error (degrees)
            "FWHM": FWHM[5], 
            "FWHM_erroe":Err_FWHM[5],
            #Value of the area under the peak peak  & error
            "Area": Area_under_peak[5],
            "Area_error":Err_Area[5]
        },
        #If your code finds more than one peak, then repeat 
        # this dictionary for each peak you find
        ], 
        #The fraction of Cu in your partiocular alloy (0.0-1.0)
        #  and the error
        "Composition":composition,
        "Composition_error":comp_err,
        #Size of the grains in your sample in nm & error
        "Grain size":final_grainsize, 
        "Grain size_error":err_grain
    }
    return results
