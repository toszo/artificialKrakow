from Main import MainClass
from MainAnn import MainAnn

def main(args=None):
    MainClass('CartPole-v0').execute()
    #MainAnn('CartPole-v0').execute()

if __name__ == "__main__":
    main()