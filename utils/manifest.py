import sys

def print_title(title, length = 100, height = 5):
    print '\n\n'
    print '*' * length
    
    for i in range(height / 2):
        print '*' + ' ' * (length - 2) + '*'
        
    print '*' + ' ' * ((length - 2 - len(title)) / 2) +\
            title.upper() +\
        ' ' * (length - 2 - len(title) - (length - 2 - len(title)) / 2) + '*'
    
    for i in range(height / 2):
        print '*' + ' ' * (length - 2) + '*'
        
    print '*' * length
    
def print_percentage(keyword, current, total, scrolling = False, length = 100):
    prefix = keyword
    percentage = current * 1.0 / total
    postfix = '|%3d' % int(percentage * 100) + '%'
    percentage_len = int((length - len(prefix) - 1 - len(postfix)) * percentage)
    printout = prefix + '|' + '=' * percentage_len + '>' + '.' * (length - len(prefix) - 2 - percentage_len - len(postfix)) + postfix
    if scrolling:
        printout = scrolling + ' ' * (length - len(scrolling)) + '\n' + printout
    sys.stdout.write('\r' + printout)
    sys.stdout.flush()
    
def print_sep(length = 100):
    print '#' * length
    
def print_gap():
    print '\n\n'
    
def print_tip(tip):
    print_gap()
    print '\t\t' + tip
    print_gap()

def print_stdout(string):
    sys.stdout.write('\r' + string)