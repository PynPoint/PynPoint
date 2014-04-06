#to manage PynPoint variables that will be stored in the PynPoint context

class im_arr_store:
    def __init__(self,im_arr):
        self.im_arr = im_arr
    def get(self):
        return self.im_arr