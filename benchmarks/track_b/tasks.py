"""
Override-tree benchmark.

A document is a bag of clauses (random order):
  RULE     id  action        : "action is permitted" (polarity fixed: permit; default is forbid)
  CARVEOUT id  target_rule   : revokes that rule
  WAIVER   id  target_carve  : revokes that carve-out (rule is restored)
  REWAIVE  id  target_waiver : revokes that waiver (carve-out active again)   -- depth 3
Query: one action. Label 1 iff the action has an *effective* permit rule.

Each clause is 3 tokens: [TYPE, ARG, ID]. Distractor chains for other actions are included,
so the model must match ids, not count clause types.

Per example we also return, for every clause, whether it is "live" (contributes to the answer)
or "revoked"/"irrelevant", which the leakage metrics use.
"""
import random, torch

N_ACTIONS, N_IDS = 6, 24
TYPES = ["RULE", "REVOKE", "QUERY", "PAD"]
T = {t: i for i, t in enumerate(TYPES)}
# token = type * (N_IDS) + arg ; ids are a separate stream
VOCAB_TYPE, VOCAB_ARG, VOCAB_ID = len(TYPES), max(2 * N_ACTIONS, N_IDS), N_IDS + 1


def make_chain(rng, action, depth, ids, polarity=None):
    """returns clauses [(type, arg, id, live)] for one action with override depth `depth`.
    RULE arg encodes action and polarity: arg = action + N_ACTIONS * polarity (1 = permit, 0 = forbid)."""
    if polarity is None: polarity = rng.randint(0, 1)
    r = ids.pop(); cl = [["RULE", action + N_ACTIONS * polarity, r, True]]
    # a uniform REVOKE chain: clause k revokes clause k-1; liveness alternates from the top
    prev = r
    for k in range(depth):
        c = ids.pop(); cl.append(["REVOKE", prev, c, True]); prev = c
    for k in range(len(cl)):               # last clause is live; each live clause kills the one it targets
        cl[k][3] = ((depth - k) % 2 == 0)
    permitted = cl[0][3] and polarity == 1   # default (no effective rule) = forbid
    return cl, permitted


def make_example(rng, depth, n_distractors=2, distractor_depths=(0, 1, 2)):
    ids = list(range(N_IDS)); rng.shuffle(ids)
    actions = list(range(N_ACTIONS)); rng.shuffle(actions)
    q_action = actions[0]
    if rng.random() < 0.15:                      # no rule at all for the queried action
        chain, label = [], False
    else:
        chain, label = make_chain(rng, q_action, depth, ids)
    for k in range(n_distractors):
        d_chain, _ = make_chain(rng, actions[1 + k], rng.choice(distractor_depths), ids)
        for c in d_chain: c[3] = None        # irrelevant to the query
        chain += d_chain
    rng.shuffle(chain)
    return chain, q_action, int(label)


def encode(chain, q_action, max_len):
    typ = [T[c[0]] for c in chain] + [T["QUERY"]]
    arg = [c[1] for c in chain] + [q_action]
    cid = [c[2] for c in chain] + [N_IDS]
    live = [(1 if c[3] else (0 if c[3] is False else -1)) for c in chain] + [1]
    while len(typ) < max_len:
        typ.append(T["PAD"]); arg.append(0); cid.append(N_IDS); live.append(-1)
    return typ, arg, cid, live


def dataset(n, depths, seed, max_len=16):
    rng = random.Random(seed)
    X_t, X_a, X_i, L, Y, D = [], [], [], [], [], []
    for k in range(n):
        d = rng.choice(depths)
        chain, qa, y = make_example(rng, d)
        t, a, i, l = encode(chain, qa, max_len)
        X_t.append(t); X_a.append(a); X_i.append(i); L.append(l); Y.append(y); D.append(d)
    return (torch.tensor(X_t), torch.tensor(X_a), torch.tensor(X_i), torch.tensor(L),
            torch.tensor(Y), torch.tensor(D))


if __name__ == "__main__":
    rng = random.Random(0)
    for d in range(4):
        ch, qa, y = make_example(rng, d)
        print(f"depth {d} query={qa} label={y}")
        for c in ch: print("   ", c)
    Xt, Xa, Xi, L, Y, D = dataset(1000, [0, 1, 2, 3], 0)
    print("label balance by depth:", {d: round(Y[D == d].float().mean().item(), 2) for d in range(4)})
