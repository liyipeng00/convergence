import pandas as pd
import logging
import os

def prcolor(info, color='white'):
    # ANSI code 31 represents standard intensity red, while ANSI code 91 represents bright intensity red.
    ansi_code_color = {'red': 91, 'green': 92, 'yellow': 93, 'blue': 94, 'magenta': 95}
    if color in ansi_code_color.keys():
        print("\033[{}m{}\033[00m" .format(ansi_code_color[color], info))
    else:
        print(info)

def logconfig(path='./save/', name:str='', flag=''):
    if 'Log' in flag:
        filepath = path+'{}.log'.format(name)
        if (os.path.exists(filepath)):
            os.remove(filepath)
        logging.basicConfig(filename=filepath, level=logging.INFO)
    else:
        pass

def add_log(info:str='', color:str='white', flag:str=''):
    if 'Print' in flag:
        prcolor(info, color)
    if 'Log' in flag:
        logging.info(info)
    pass

def record_exp_result(filename, result):
    savepath = './save/'
    filepath = '{}/{}.csv'.format(savepath, filename)
    if result['round'] == 0:
        if (os.path.exists(filepath)):
            os.remove(filepath)
        with open (filepath, 'a+') as f:
            f.write('{},{},{},{},{},{},{}\n'.format('round', 
                                                    'train_loss', 'train_top1',  'train_top5', 
                                                    'test_loss',  'test_top1',   'test_top5'))
    else:
        with open (filepath, 'a+') as f:
            f.write('{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'
                    .format(result['round'],
                            result['train_loss'], result['train_top1'], result['train_top5'], 
                            result['test_loss'],  result['test_top1'],  result['test_top5']))

def record_exp_result2(filename, result):
    savepath = './save/'
    filepath = '{}/{}.csv'.format(savepath, filename)
    if result['round'] == 0:
        if (os.path.exists(filepath)):
            os.remove(filepath)
        with open (filepath, 'a+') as f:
            f.write('{},{},{},{},{},{},{},{},{},{}\n'.format('round', 
                                                             'train_loss',  'train_top1',   'train_top5', 
                                                             'train2_loss', 'train2_top1',  'train2_top5', 
                                                             'test_loss',   'test_top1',    'test_top5'))
    else:
        with open (filepath, 'a+') as f:
            f.write('{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'
                    .format(result['round'],
                            result['train_loss'],  result['train_top1'],  result['train_top5'],
                            result['train2_loss'], result['train2_top1'], result['train2_top5'],
                            result['test_loss'],   result['test_top1'],   result['test_top5']))

#Depreciated. Replace it with record_exp_result
# def record_data(): 
#     global args, train_loss, val_loss, train_tops, val_tops

#     filename = record_setup(args)
#     record_tocsv(name='{}'.format(filename), path='./save/',
#         train_loss=train_loss, val_loss=val_loss, 
#         train_top1=train_tops[0], val_top1=val_tops[0], 
#         train_top5=train_tops[1], val_top5=val_tops[1])


def record_tocsv(name, path='./save/', **kwargs):
    #epoch = [i for i in range(1, len(kwargs[list(kwargs.keys())[0]])+1)]
    #df = pd.DataFrame(kwargs)  
    #df = pd.DataFrame.from_dict(kwargs, orient='index')   
    df = pd.DataFrame(pd.DataFrame.from_dict(kwargs, orient='index').values.T, columns=list(kwargs.keys()))
    file_name = path + name + ".csv"    
    df.to_csv(file_name, index = False)

def read_fromcsv(name, path='./save/'):
    if '.csv' in name:
        df = pd.read_csv("{}{}".format(path, name))
    else:
        df = pd.read_csv("{}{}.csv".format(path, name))
    return df

def record_toexcel(name, **kwargs):
    path = './save/'
    #epoch = [i for i in range(1, len(kwargs[list(kwargs.keys())[0]])+1)]
    df = pd.DataFrame(kwargs)     
    file_name = path + name + ".xls"    
    df.to_excel(file_name, sheet_name= "Sheet1", index = False) 

def read_fromexcel(name):
    path = './save/'
    df = pd.read_excel("{}{}.xls".format(path, name), sheet_name="Sheet1") # Sheet1
    return df

def exceltocsv():
    path = './save/'
    name='client alex cifar layer 1'
    df = pd.read_excel("{}{}.xls".format(path, name)) # Sheet1
    df.to_csv("{}{}.csv".format(path, name), index = False)

if __name__ == '__main__':
    exceltocsv()
    


