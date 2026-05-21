import matplotlib
matplotlib.use("TkAgg")

from utils.app import App

# Entrypoint
if __name__ == "__main__":
    app = App()
    app.mainloop()