from knlm import *
import sys

if len(sys.argv) < 2:
    print("Usage: python -m knlm [modelpath]")
    exit()
print('Loading %s ...' % sys.argv[1])
mdl = KneserNey.load(sys.argv[1])
print('Order: %d, Vocab Size: %d, Vocab Width: %d' % (mdl.order, mdl.vocabs, mdl._wsize))

while 1:
    sent = input('>> ').strip().split()
    if sent:
        ll = mdl.evaluateSent(sent)
        print("ll: %g, avg ll: &g" % (ll, ll / (len(sent) + 1)))
        for c, l in zip(sent, mdl.evaluateEachWord(sent)):
            print('%s: %g' % (c, l))
