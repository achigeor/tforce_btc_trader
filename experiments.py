def baseline(**kwargs):
    b = dict(baseline=dict(
        type="mlp",
        sizes=[128, 128],
        epochs=5,
        update_batch_size=128,
        learning_rate=.01
    ))
    b['baseline'].update(**kwargs)
    return b


def network(arch, n=256, d=.2, a='elu'):
    net = [dict(type='dropout', size=n, dropout=d)] if d else []
    for layer in arch:
        if layer == 'L':
            net.append(dict(type='lstm', size=n, dropout=d))
        elif layer == 'D':
            net.append(dict(type='dense2', size=n, dropout=d, activation=a))
    return dict(network=net)

confs = [
    dict(k='main', v=[dict(k='-', v=dict())]),
    dict(k='activation', v=[
        dict(k='selu', v=network('LLDD', a='selu')),
        dict(k='tanh', v=network('LLDD', a='tanh'))
    ]),
    dict(k='dropout', v=[
        dict(k='l2_reg', v=network('LLDD', d=None)),
        dict(k='.1', v=network('LLDD', d=.1)),
        dict(k='.5', v=network('LLDD', d=.5))
    ]),
    dict(k='baseline', v=[  # loser
        dict(k='main', v=baseline()),
        dict(k='2x64', v=baseline(sizes=[64, 64])),
        dict(k='2x64', v=baseline(sizes=[128, 128])),
        dict(k='2x256', v=baseline(sizes=[256, 256])),
        dict(k='epochs10', v=baseline(epochs=10)),
        dict(k='update_batch_size64', v=baseline(update_batch_size=64)),
        dict(k='update_batch_size1024', v=baseline(update_batch_size=1024)),
        dict(k='learning_rate.001', v=baseline(learning_rate=.001)),
    ]),
    dict(k='network', v=[
        dict(k=f'{arch}.{neur}', v=network(arch, neur))
        for neur in [640, 512, 256, 128, 64]
        for arch in [
            'L',
            'DL', 'LD', 'LL',
            'DLD', 'DLL', 'LDD', 'LLD', 'LLL',
            'DLLD', 'LLDD', 'LLLD',
            'DLLLD'
        ] # Good were DLD, DLLD, LLDD(winner)
    ]),
    dict(k='batch', v=[
        dict(k='b128.o64', v=dict(batch_size=128, optimizer_batch_size=64)),
        dict(k='b256.o128', v=dict(batch_size=256, optimizer_batch_size=128)),
        dict(k='b1024.o128', v=dict(batch_size=1024, optimizer_batch_size=128)),
        dict(k='b1024.o512', v=dict(batch_size=1024, optimizer_batch_size=512)),
        dict(k='b4096.o128', v=dict(batch_size=4096, optimizer_batch_size=128)),
        dict(k='b4096.o1024', v=dict(batch_size=4096, optimizer_batch_size=1024)),
        dict(k='b4096.o2048', v=dict(batch_size=4096, optimizer_batch_size=2048)),
    ]),
    dict(k='epochs', v=[
        dict(k='1', v=dict(epochs=1)),
        dict(k='10', v=dict(epochs=10)),
        dict(k='40', v=dict(epochs=40)),  # loser
    ]),
    dict(k='gae_rewards', v=[
        dict(k='True', v=dict(gae_rewards=True)),  # winner=False
    ]),
    dict(k='keep_last', v=[
        dict(k='True', v=dict(keep_last=True)),  # unclear, winner~=False
    ]),
    dict(k='random_sampling', v=[
        dict(k='False', v=dict(random_sampling=False)),  # unclear
    ]),
    dict(k='discount', v=[
        dict(k='.95', v=dict(discount=.95)),
        dict(k='.97', v=dict(discount=.97)),
    ]),
    dict(k='learning_rate', v=[
        dict(k='.01', v=dict(learning_rate=.01)),
        dict(k='.0001', v=dict(learning_rate=.0001)),
    ]),
    dict(k='optimizer', v=[
        dict(k='adam', v=dict(optimizer='adam')),
    ]),
]

confs = [
    dict(
        name=c['k'] + ':' + permu['k'],
        conf=permu['v']
    )
    for c in confs for permu in c['v']
]