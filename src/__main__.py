from sys import exit

try:
    import fire
    from src.cli import RagCLI
except(ImportError, KeyboardInterrupt, NameError,Exception )as e:
        print(f"\n{e}error has occured")
        exit(1)

if __name__ == "__main__":

    try:
        fire.Fire(RagCLI)
    except(ImportError, KeyboardInterrupt, NameError,Exception )as e:
        print(e)