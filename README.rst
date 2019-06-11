knlm
----

Modified Kneser-Ney smoothing language model module for Python

Installation
------------
::

    $ pip install knlm
    $ pip3 install knlm


Example
-------
::

    from knlm import KneserNey
    
    mode = 'build'
    if mode == 'build':
        # build model from corpus text. order = 3, word size = 4 byte
        mdl = KneserNey(3, 4)
        for line in open('corpus.txt', encoding='utf-8'):
            mdl.train(line.lower().strip().split())
        mdl.optimize()
        mdl.save('language.model')
    else:
        # load model from binary file
        mdl = KneserNey.load('language.model')
        print('Loaded')
    print('Order: %d, Vocab Size: %d, Vocab Width: %d' % (mdl.order, mdl.vocabs, mdl._wsize))

    # evaluate sentence score
    print(mdl.evaluateSent('I love kiwi .'.split()))
    print(mdl.evaluateSent('ego kiwi amo .'.split()))
    
    # evaluate scores for each word
    print(mdl.evaluateEachWord('I love kiwi .'.split()))
    print(mdl.evaluateEachWord('ego kiwi amo .'.split()))
