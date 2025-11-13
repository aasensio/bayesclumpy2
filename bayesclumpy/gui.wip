import numpy as np
import PySimpleGUI as sg
import matplotlib.pyplot as pl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# VARS CONSTS:
_VARS = {'window': False,
        'fig_agg': False,
        'pltFig': False,
        'dataSize': 60}

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def makeSynthData():
    xData = np.random.randint(100, size=_VARS['dataSize'])
    yData = np.linspace(0, _VARS['dataSize'],
                        num=_VARS['dataSize'], dtype=int)
    return (xData, yData)

def drawChart():
    _VARS['pltFig'] = pl.figure()
    dataXY = makeSynthData()
    pl.plot(dataXY[0], dataXY[1], '.k')
    _VARS['fig_agg'] = draw_figure(
        _VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])

def updateChart():
    _VARS['fig_agg'].get_tk_widget().forget()
    dataXY = makeSynthData()
    # plt.cla()
    pl.clf()
    pl.plot(dataXY[0], dataXY[1], '.k')
    _VARS['fig_agg'] = draw_figure(
        _VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])

def updateData(val):
    _VARS['dataSize'] = val
    updateChart()

def inspect(samples):
    AppFont = 'Any 16'
    SliderFont = 'Any 14'
    sg.theme('black')
    
    layout_figs = [sg.Canvas(key='figCanvas', background_color='#FDF6E3'),
                sg.Canvas(key='figCanvas2', background_color='#FDF6E3')]

    frame_Y = sg.Frame('Y', [[sg.Text('min')], [sg.Slider(range=(5, 100), orientation='h', size=(20, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='slidermin_Y',
                     enable_events=True)],
                    [sg.Text('max')], [sg.Slider(range=(5, 100), orientation='h', size=(20, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='slidermax_Y',
                     enable_events=True)],
                     [sg.Radio('Uniform', "Y_PRIOR", default=True)],
                     [sg.Radio('Gaussian', "Y_PRIOR")],
                     [sg.Radio('Dirac', "Y_PRIOR")]])
    
    frame_sigma = sg.Frame(u'\u03C3', [[sg.Text('min')], [sg.Slider(range=(15, 70), orientation='h', size=(20, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='slidermin_sigma',
                     enable_events=True)],
                     [sg.Text('max')], [sg.Slider(range=(15, 70), orientation='h', size=(20, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='slidermax_sigma',
                     enable_events=True)],
                     [sg.Radio('Uniform', "sigma_PRIOR", default=True)],
                     [sg.Radio('Gaussian', "sigma_PRIOR")],
                     [sg.Radio('Dirac', "sigma_PRIOR")]])

    frame_N = sg.Frame('N', [[sg.Text('min')], [sg.Slider(range=(1, 15), orientation='h', size=(20, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='slidermin_N',
                     enable_events=True)],
                    [sg.Text('max')], [sg.Slider(range=(1, 15), orientation='h', size=(20, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='slidermax_N',
                     enable_events=True)],
                     [sg.Radio('Uniform', "N_PRIOR", default=True)],
                     [sg.Radio('Gaussian', "N_PRIOR")],
                     [sg.Radio('Dirac', "N_PRIOR")]])

    frame_q = sg.Frame('q', [[sg.Text('min')], [sg.Slider(range=(0, 3), orientation='h', size=(20, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='slidermin_q',
                     enable_events=True)],
                    [sg.Text('max')], [sg.Slider(range=(0, 3), orientation='h', size=(20, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='slidermax_q',
                     enable_events=True)],
                     [sg.Radio('Uniform', "q_PRIOR", default=True)],
                     [sg.Radio('Gaussian', "q_PRIOR")],
                     [sg.Radio('Dirac', "q_PRIOR")]])

    frame_tau = sg.Frame(u'\u03C4', [[sg.Text('min')], [sg.Slider(range=(10, 300), orientation='h', size=(20, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='slidermin_tau',
                     enable_events=True)],
                    [sg.Text('max')], [sg.Slider(range=(10, 300), orientation='h', size=(20, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='slidermax_tau',
                     enable_events=True)],
                     [sg.Radio('Uniform', "tau_PRIOR", default=True)],
                     [sg.Radio('Gaussian', "tau_PRIOR")],
                     [sg.Radio('Dirac', "tau_PRIOR")]])

    frame_i = sg.Frame('i', [[sg.Text('min')], [sg.Slider(range=(0, 90), orientation='h', size=(20, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='slidermin_i',
                     enable_events=True)],
                    [sg.Text('max')], [sg.Slider(range=(0, 90), orientation='h', size=(20, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='slidermax_i',
                     enable_events=True)],
                     [sg.Radio('Uniform', "i_PRIOR", default=True)],
                     [sg.Radio('Gaussian', "i_PRIOR")],
                     [sg.Radio('Dirac', "i_PRIOR")]])

    frame_shift = sg.Frame('shift', [[sg.Text('min')], [sg.Slider(range=(-3, 3), orientation='h', size=(20, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='slidermin_shift',
                     enable_events=True)],
                    [sg.Text('max')], [sg.Slider(range=(-3, 3), orientation='h', size=(20, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='slidermax_shift',
                     enable_events=True)],
                     [sg.Radio('Uniform', "shift_PRIOR", default=True)],
                     [sg.Radio('Gaussian', "shift_PRIOR")],
                     [sg.Radio('Dirac', "shift_PRIOR")]])

    frame_Av = sg.Frame('Av', [[sg.Text('min')], [sg.Slider(range=(0, 300), orientation='h', size=(20, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='slidermin_Av',
                     enable_events=True)],
                    [sg.Text('max')], [sg.Slider(range=(0, 300), orientation='h', size=(20, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='slidermax_Av',
                     enable_events=True)],
                     [sg.Radio('Uniform', "Av_PRIOR", default=True)],
                     [sg.Radio('Gaussian', "Av_PRIOR")],
                     [sg.Radio('Dirac', "Av_PRIOR")]])
    
    frame_z = sg.Frame('z', [[sg.Text('min')], [sg.Slider(range=(0, 6), orientation='h', size=(20, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='slidermin_z',
                     enable_events=True)],
                    [sg.Text('max')], [sg.Slider(range=(0, 6), orientation='h', size=(20, 20),
                     default_value=_VARS['dataSize'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='slidermax_z',
                     enable_events=True)],
                     [sg.Radio('Uniform', "z_PRIOR", default=True)],
                     [sg.Radio('Gaussian', "z_PRIOR")],
                     [sg.Radio('Dirac', "z_PRIOR")]])

    layout1 = [frame_Y, frame_sigma, frame_N, frame_tau, frame_i, frame_shift, frame_Av, frame_z]

    layout = [layout_figs, layout1]
    # layout = [,
    #       [sg.Text(text="Random sample size :",
    #                font=SliderFont,
    #                background_color='#FDF6E3',
    #                pad=((0, 0), (10, 0)),
    #                text_color='Black'),
    #        sg.Slider(range=(4, 1000), orientation='h', size=(34, 20),
    #                  default_value=_VARS['dataSize'],
    #                  background_color='#FDF6E3',
    #                  text_color='Black',
    #                  key='-Slider-',
    #                  enable_events=True),
    #        sg.Button('Resample',
    #                  font=AppFont,
    #                  pad=((4, 0), (10, 0)))],
    #       # pad ((left, right), (top, bottom))
    #       [sg.Button('Exit', font=AppFont, pad=((540, 0), (0, 0)))]]

    _VARS['window'] = sg.Window('Such Window',
                                layout,
                                finalize=True,
                                resizable=True,
                                element_justification="center")

    drawChart()

    while True:
        event, values = _VARS['window'].read(timeout=200)        
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == 'Resample':        
            updateChart()
        elif event == '-Slider-':
            updateData(int(values['-Slider-']))
        

    _VARS['window'].close()