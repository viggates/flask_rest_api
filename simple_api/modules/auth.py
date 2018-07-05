class Authentication(object):
    def authenticate(self):
        f = open('centrify_fake/modules/authdata.xml' ,'r')
        return f.read()
