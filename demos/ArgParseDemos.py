# -*- encoding: utf-8 -*-
# -------------------------------------------------------------------------------
'''

@File      :  ArgParseDemos.py  
@Author    :  Qin ZhaoYu   

@Desc      :  None  

Change History:
---------------------------------------------------------------------------------
version1.0.0, 2022/02/06, Qin ZhaoYu
Initial commit.
---------------------------------------------------------------------------------
'''
# -------------------------------------------------------------------------------
import argparse
import sys


def demo1():
    '''默认参数只有可选参数 -help 与其短写 -h，无法接受命令行参数。

    返回值是 Namespace 类型，无迭代器。
    '''
    brief = "==*== usage of default argument ==*=="
    parser = argparse.ArgumentParser(description=brief)

    args = parser.parse_args()
    return args


def demo2():
    '''解析位置参数，参数名称不带'-'。
    '''
    brief = "==*== usage of palcement argument ==*=="
    parser = argparse.ArgumentParser(description=brief)

    parser.add_argument('e', help="placement arg")

    args = parser.parse_args()
    return args


def demo3():
    '''可选参数，参数名称带'-' 或 '--'。
    '''
    brief = "==*== usage of optional argument ==*=="
    parser = argparse.ArgumentParser(description=brief)

    parser.add_argument('-v', '--verbosity', help="optional arg")
    parser.add_argument('e', help="placement arg")

    args = parser.parse_args()
    return args.verbosity, args.e  # 取值方式，大小写敏感


def demo4():
    '''参数描述和限制。
    '''
    brief = "==*== usage of argument constraints ==*=="
    parser = argparse.ArgumentParser(description=brief)

    parser.add_argument('-v', '--verbosity',
                        choices=['yes', 'no', 'fuck'],
                        default='fuck',
                        help="optional arg")
    parser.add_argument('e', type=int, help="placement arg")

    args = parser.parse_args()
    return args.verbosity, args.e


def demo5():
    '''添加互斥参数。
    '''
    brief = "==*== usage of mutually exclusive argument ==*=="
    parser = argparse.ArgumentParser(description=brief)

    parser.add_argument('-v', '--verbosity',
                        choices=['yes', 'no', 'fuck'],
                        default='fuck',
                        help="optional arg")
    parser.add_argument('e', type=int, help="placement arg")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', '--car', action="store_true",
                       help="driving home")
    group.add_argument('-f', '--foot',
                       action="store_true", help='walking home')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("args from sys: ", sys.argv)

    # args = demo1()
    # print("args from argparse: ", args)

    # args = demo2()
    # print("args from argparse: ", args)

    # args = demo3()
    # print("args from argparse: ", args)

    # args = demo4()
    # print("args from argparse: ", args)

    args = demo5()
    print("args from argparse: ", args)
