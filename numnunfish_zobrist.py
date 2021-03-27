#!/usr/bin/env pypy
# -*- coding: utf-8 -*-
import os
# os.environ['NUMBA_DISABLE_JIT'] = "1"
import time
from collections import namedtuple
from numba import njit, typed
from numba.experimental import jitclass
from numba.core import types
import numba as nb
import numpy as np

###############################################################################
# Piece-Square tables. Tune these to change sunfish's behaviour
###############################################################################

PADDING = 0
EMPTY = 1

PAWN = 2
KNIGHT = 3
BISHOP = 4
ROOK = 5
QUEEN = 6
KING = 7

BLACK_OFFSET = KING - PAWN + 1

BLACK_PAWN = PAWN + BLACK_OFFSET
BLACK_KNIGHT = KNIGHT + BLACK_OFFSET
BLACK_BISHOP = BISHOP + BLACK_OFFSET
BLACK_ROOK = ROOK + BLACK_OFFSET
BLACK_QUEEN = QUEEN + BLACK_OFFSET
BLACK_KING = KING + BLACK_OFFSET


piece = { PAWN: 100, KNIGHT: 280, BISHOP: 320, ROOK: 479, QUEEN: 929, KING: 60000 }
pst = {
    PAWN: (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    KNIGHT: ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    BISHOP: ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    ROOK: (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    QUEEN: (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    KING: (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}
# Pad tables and join piece and pst dictionaries
for k, table in pst.items():
    padrow = lambda row: (0,) + tuple(x+piece[k] for x in row) + (0,)
    pst[k] = sum((padrow(table[i*8:i*8+8]) for i in range(8)), ())
    pst[k] = (0,)*20 + pst[k] + (0,)*20

# Make numpy typed:
pst_tup = pst
pst = np.empty((KING + 1, 120), dtype=np.int64)
for k in pst_tup:
    pst[k, :] = np.array(pst_tup[k], dtype='i8')


###############################################################################
# Global constants
###############################################################################

# Our board is represented as a 120 character string. The padding allows for
# fast detection of moves that don't stay within the board.
A1, H1, A8, H8 = 91, 98, 21, 28
initial_str = (
    '         \n'  #   0 -  9
    '         \n'  #  10 - 19
    ' rnbqkbnr\n'  #  20 - 29
    ' pppppppp\n'  #  30 - 39
    ' ........\n'  #  40 - 49
    ' ........\n'  #  50 - 59
    ' ........\n'  #  60 - 69
    ' ........\n'  #  70 - 79
    ' PPPPPPPP\n'  #  80 - 89
    ' RNBQKBNR\n'  #  90 - 99
    '         \n'  # 100 -109
    '         \n'  # 110 -119
)
trans = {' ': PADDING, '\n': PADDING, 'R': ROOK, 'N': KNIGHT, 'B': BISHOP, 'Q': QUEEN, 'K': KING, 'P': PAWN, '.': EMPTY}
initial = np.zeros(len(initial_str), dtype=np.int64)
for i in range(len(initial_str)):
    x = initial_str[i]
    initial[i] = trans[x] if x in trans else (trans[x.upper()] + BLACK_OFFSET)

# Lists of possible moves for each piece type.
N, E, S, W, STOP = -10, 1, 10, -1, 100
directions_d = {}
directions_d[PAWN] = np.array([N, N+N, N+W, N+E, STOP], dtype='i8')
directions_d[KNIGHT] = np.array([N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W, STOP], dtype='i8')
directions_d[BISHOP] = np.array([N+E, S+E, S+W, N+W, STOP], dtype='i8')
directions_d[ROOK] = np.array([N, E, S, W, STOP], dtype='i8')
directions_d[QUEEN] = np.array([N, E, S, W, N+E, S+E, S+W, N+W, STOP], dtype='i8')
directions_d[KING] = np.array([N, E, S, W, N+E, S+E, S+W, N+W, STOP], dtype='i8')
directions = np.empty((KING + 1, 10), dtype=np.int64)
for k in directions_d:
    directions[k, :len(directions_d[k])] = directions_d[k]

# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
# When a MATE is detected, we'll set the score to MATE_UPPER - plies to get there
# E.g. Mate in 3 will be MATE_UPPER - 6
MATE_LOWER = types.int64(piece[KING] - 10*piece[QUEEN])
MATE_UPPER = types.int64(piece[KING] + 10*piece[QUEEN])

# The table size is the maximum number of elements in the transposition table.
TABLE_SIZE = 1e7

# Constants for tuning search
QS_LIMIT = 219
EVAL_ROUGHNESS = 13
DRAW_TEST = True

zobrist = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(len(initial), BLACK_KING), dtype=np.int64)

@njit
def zhash(board):
    h = np.int64(0)
    for i in range(21, 100):
        p = board[i]
        if p >= PAWN:
            h = np.bitwise_xor(h, zobrist[i - 21, p - PAWN])
    return h


zobrist_score = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max - 1, size=4 * MATE_UPPER, dtype=np.int64)
zobrist_extra = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max - 1, size=1000, dtype=np.int64)
@njit
def zhash_full(board, score, wc, bc, ep, kp):
    h = zhash(board)
    h = np.bitwise_xor(h, zobrist_score[MATE_UPPER + int(score) + 1])
    h = np.bitwise_xor(h, zobrist_extra[int(wc[0])])
    h = np.bitwise_xor(h, zobrist_extra[100 + int(wc[1])])
    h = np.bitwise_xor(h, zobrist_extra[200 + int(bc[0])])
    h = np.bitwise_xor(h, zobrist_extra[300 + int(bc[1])])
    h = np.bitwise_xor(h, zobrist_extra[400 + ep])
    h = np.bitwise_xor(h, zobrist_extra[500 + kp])
    return h


@njit
def put(board, i, p):
    board = board.copy()
    board[i] = p
    return board


@njit
def iswhite(p):
    return PAWN <= p <= KING


@njit
def isblack(p):
    return p >= BLACK_PAWN


@njit
def swapwhiteblack(board):
    return board + BLACK_OFFSET * np.logical_and(PAWN <= board, board <= KING) - BLACK_OFFSET * (board >= BLACK_PAWN)


###############################################################################
# Chess logic
###############################################################################

@jitclass([
    ('board', nb.typeof(initial)),
    ('score', types.int64),
    ('wc', nb.typeof((1, 0))),
    ('bc', nb.typeof((1, 0))),
    ('ep', types.int64),
    ('kp', types.int64)
])
class Position:
    """ A state of a chess game
    board -- a 120 char representation of the board
    score -- the board evaluation
    wc -- the castling rights, [west/queen side, east/king side]
    bc -- the opponent castling rights, [west/king side, east/queen side]
    ep - the en passant square
    kp - the king passant square
    """

    def __init__(self, board, score, wc, bc, ep, kp):
        self.board = board
        self.score = int(score)
        self.wc = wc
        self.bc = bc
        self.ep = int(ep)
        self.kp = int(kp)

    def string_board(self):
        return "".join([str(x) for x in self.board])

    @property
    def str(self):
        return zhash_full(self.board, self.score, self.wc, self.bc, self.ep, self.kp)


    def gen_moves(self, directions):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
        for i, p in enumerate(self.board):
            if not iswhite(p): continue
            for d in directions[p]:
                if d == STOP:
                    break
                j = i
                while True:
                    j += d
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q == PADDING or iswhite(q): break
                    # Pawn move, double move and capture
                    if p == PAWN and d in (N, N+N) and q != EMPTY: break
                    if p == PAWN and d == N+N and (i < A1+N or self.board[i+N] != EMPTY): break
                    if p == PAWN and d in (N+W, N+E) and q == EMPTY \
                            and j not in (self.ep, self.kp, self.kp-1, self.kp+1): break
                    # Move it
                    yield (i, j)
                    # Stop crawlers from sliding, and sliding after captures
                    if (p == PAWN or p == KNIGHT or p == KING) or isblack(q): break
                    # Castling, by sliding the rook next to the king
                    if i == A1 and self.board[j+E] == KING and self.wc[0]: yield (j+E, j+W)
                    if i == H1 and self.board[j+W] == KING and self.wc[1]: yield (j+W, j+E)

    def rotate(self):
        ''' Rotates the board, preserving enpassant '''
        return Position(
            swapwhiteblack(self.board[::-1]), -self.score, self.bc, self.wc,
            119-self.ep if self.ep else 0,
            119-self.kp if self.kp else 0)

    def nullmove(self):
        ''' Like rotate, but clears ep and kp '''
        return Position(
            swapwhiteblack(self.board[::-1]), -self.score,
            self.bc, self.wc, 0, 0)

    def move(self, move, pst):
        i, j = move
        p, q = self.board[i], self.board[j]
        # Copy variables and reset ep and kp
        board = self.board.copy()
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        score = self.score + self.value(move, pst)
        # Actual move
        board = put(board, j, board[i])
        board = put(board, i, EMPTY)
        # Castling rights, we move the rook or capture the opponent's
        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])
        # Castling
        if p == KING:
            wc = (False, False)
            if abs(j-i) == 2:
                kp = (i+j)//2
                board = put(board, A1 if j < i else H1, EMPTY)
                board = put(board, kp, ROOK)
        # Pawn promotion, double move and en passant capture
        if p == PAWN:
            if A8 <= j <= H8:
                board = put(board, j, QUEEN)
            if j - i == 2*N:
                ep = i + N
            if j == self.ep:
                board = put(board, j+S, EMPTY)
        # We rotate the returned position, so it's ready for the next player
        return Position(board, score, wc, bc, ep, kp).rotate()

    def value(self, move, pst):
        i, j = move
        p, q = self.board[i], self.board[j]
        # Actual move
        score = pst[p][j] - pst[p][i]
        # Capture
        if isblack(q):
            score += pst[q - BLACK_OFFSET][119-j]
        # Castling check detection
        if abs(j-self.kp) < 2:
            score += pst[KING][119-j]
        # Castling
        if p == KING and abs(i-j) == 2:
            score += pst[ROOK][(i+j)//2]
            score -= pst[ROOK][A1 if j < i else H1]
        # Special pawn stuff
        if p == PAWN:
            if A8 <= j <= H8:
                score += pst[QUEEN][j] - pst[PAWN][j]
            if j == self.ep:
                score += pst[PAWN][119-(j+S)]
        return score

###############################################################################
# Search logic
###############################################################################

# lower <= s(pos) <= upper
Entry = namedtuple('Entry', 'lower upper')
NoneMove = (0, 0)

class Searcher:
    def __init__(self):
        self.tp_score = typed.Dict.empty(
            key_type=types.int64,
            value_type=nb.typeof(Entry(0, 0)),
        )
        self.tp_move = typed.Dict.empty(
            key_type=types.int64,
            value_type=nb.typeof(NoneMove),
        )
        self.history = typed.Dict.empty(
            key_type=types.int64,
            value_type=types.int64,
        )

    def search(self, pos, history=None):
        yield from searcher_search(pos, history, self.tp_score, self.tp_move, pst, directions)


@njit
def tup_to_str(tup):
    return np.bitwise_xor(np.bitwise_xor(tup[0], zobrist_extra[600 + int(tup[1])]), zobrist_extra[700 + int(tup[1])])


@njit
def searcher_search(pos, history, tp_score, tp_move, pst, directions):
    """ Iterative deepening MTD-bi search """
    if DRAW_TEST:
        # print('# Clearing table due to new history')
        tp_score.clear()

    # In finished games, we could potentially go far enough to cause a recursion
    # limit exception. Hence we bound the ply.
    for depth in range(1, 1000):
        # The inner loop is a binary search on the score of the position.
        # Inv: lower <= score <= upper
        # 'while lower != upper' would work, but play tests show a margin of 20 plays
        # better.
        lower, upper = -MATE_UPPER, MATE_UPPER
        while lower < upper - EVAL_ROUGHNESS:
            gamma = (lower+upper+1)//2
            score = searcher_bound(pos, gamma, depth, history, tp_move, tp_score, pst, directions)
            if score >= gamma:
                lower = score
            if score < gamma:
                upper = score
        # We want to make sure the move to play hasn't been kicked out of the table,
        # So we make another call that must always fail high and thus produce a move.
        searcher_bound(pos, lower, depth, history, tp_move, tp_score, pst, directions)
        # If the game hasn't finished we can retrieve our move from the
        # transposition table.
        h = pos.str
        yield depth, tp_move.get(h, NoneMove), tp_score[tup_to_str((h, depth, True))].lower


@njit
def check_if_RBNQ(board):
    for c in board:
        if c == ROOK or c == BISHOP or c == QUEEN or c == KNIGHT:
            return True
    return False


@njit
def is_dead(pos, pst, directions):
    for m in pos.gen_moves(directions):
        if pos.value(m, pst) >= MATE_LOWER:
            return True
    return False


@njit
def all_dead(pos, pst, directions):
    for m in pos.gen_moves(directions):
        if not is_dead(pos.move(m, pst), pst, directions):
            return False
    return True


# Generator of moves to search in order.
# This allows us to define the moves, but only calculate them if needed.
@njit
def moves(pos, depth, root, gamma, history, tp_move, tp_score, pst, directions):
    # First try not moving at all. We only do this if there is at least one major
    # piece left on the board, since otherwise zugzwangs are too dangerous.
    if depth > 0 and not root and check_if_RBNQ(pos.board):
        yield NoneMove, -searcher_bound(pos.nullmove(), 1-gamma, depth-3, history, tp_move, tp_score, pst, directions, False)
    # For QSearch we have a different kind of null-move, namely we can just stop
    # and not capture anything else.
    if depth == 0:
        yield NoneMove, pos.score
    # Then killer move. We search it twice, but the tp will fix things for us.
    # Note, we don't have to check for legality, since we've already done it
    # before. Also note that in QS the killer must be a capture, otherwise we
    # will be non deterministic.

    h = pos.str
    if h in tp_move:
        killer = tp_move[h]
    else:
        killer = NoneMove
    if not (killer[0] == NoneMove[0] and killer[1] == NoneMove[1]) and (depth > 0 or pos.value(killer, pst) >= QS_LIMIT):
        yield killer, -searcher_bound(pos.move(killer, pst), 1-gamma, depth-1, history, tp_move, tp_score, pst, directions, False)

    # Then all the other moves
    remaining_moves = list(pos.gen_moves(directions))
    scores = sorted([(pos.value(m, pst), -i) for i, m in enumerate(remaining_moves)], reverse=True)
    for _, i in scores:
        move = remaining_moves[-i]  # -i is a hack to ensure same sort order as original code
        # If depth == 0 we only try moves with high intrinsic score (captures and
        # promotions). Otherwise we do all moves.
        if depth > 0 or pos.value(move, pst) >= QS_LIMIT:
            yield move, -searcher_bound(pos.move(move, pst), 1-gamma, depth-1, history, tp_move, tp_score, pst, directions, False)


@njit
def searcher_bound(pos, gamma, depth, history, tp_move, tp_score, pst, directions, root=True):
    """ returns r where
            s(pos) <= r < gamma    if gamma > s(pos)
            gamma <= r <= s(pos)   if gamma <= s(pos)"""

    # Depth <= 0 is QSearch. Here any position is searched as deeply as is needed for
    # calmness, and from this point on there is no difference in behaviour depending on
    # depth, so so there is no reason to keep different depths in the transposition table.
    if depth < 0:
        depth = 0

    # Sunfish is a king-capture engine, so we should always check if we
    # still have a king. Notice since this is the only termination check,
    # the remaining code has to be comfortable with being mated, stalemated
    # or able to capture the opponent king.
    if pos.score <= -MATE_LOWER:
        return -MATE_UPPER

    # We detect 3-fold captures by comparing against previously
    # _actually played_ positions.
    # Note that we need to do this before we look in the table, as the
    # position may have been previously reached with a different score.
    # This is what prevents a search instability.
    # FIXME: This is not true, since other positions will be affected by
    # the new values for all the drawn positions.
    h = pos.str
    if DRAW_TEST:
        if not root and h in history:
            return 0

    # Look in the table if we have already searched this position before.
    # We also need to be sure, that the stored search was over the same
    # nodes as the current search.
    x = tup_to_str((h, depth, root))
    if x in tp_score:
        entry = tp_score[x]
    else:
        entry = Entry(-MATE_UPPER, MATE_UPPER)
    if entry.lower >= gamma and (not root or tp_move.get(h, NoneMove) != NoneMove):
        return entry.lower
    if entry.upper < gamma:
        return entry.upper

    # Here extensions may be added
    # Such as 'if in_check: depth += 1'

    # Run through the moves, shortcutting when possible
    best = -MATE_UPPER
    for move, score in moves(pos, depth, root, gamma, history, tp_move, tp_score, pst, directions):
        best = max(best, score)
        if best >= gamma:
            # Clear before setting, so we always have a value
            if len(tp_move) > TABLE_SIZE: tp_move.clear()
            # Save the move for pv construction and killer heuristic
            tp_move[h] = move
            break

    # Stalemate checking is a bit tricky: Say we failed low, because
    # we can't (legally) move and so the (real) score is -infty.
    # At the next depth we are allowed to just return r, -infty <= r < gamma,
    # which is normally fine.
    # However, what if gamma = -10 and we don't have any legal moves?
    # Then the score is actaully a draw and we should fail high!
    # Thus, if best < gamma and best < 0 we need to double check what we are doing.
    # This doesn't prevent sunfish from making a move that results in stalemate,
    # but only if depth == 1, so that's probably fair enough.
    # (Btw, at depth 1 we can also mate without realizing.)
    if best < gamma and best < 0 and depth > 0:
        if all_dead(pos, pst, directions):
            in_check = is_dead(pos.nullmove(), pst, directions)
            best = -MATE_UPPER if in_check else 0

    # Clear before setting, so we always have a value
    if len(tp_score) > TABLE_SIZE: tp_score.clear()
    # Table part 2
    if best >= gamma:
        tp_score[x] = Entry(best, entry.upper)
    if best < gamma:
        tp_score[x] = Entry(entry.lower, best)

    return best


###############################################################################
# User interface
###############################################################################


def parse(c):
    fil, rank = ord(c[0]) - ord('a'), int(c[1]) - 1
    return A1 + fil - 10*rank


def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord('a')) + str(-rank + 1)


def print_pos(pos):
    print()
    uni_pieces = {ROOK:'♜', KNIGHT:'♞', BISHOP:'♝', QUEEN:'♛', KING:'♚', PAWN:'♟',
                  BLACK_ROOK:'♖', BLACK_KNIGHT:'♘', BLACK_BISHOP:'♗', BLACK_QUEEN:'♕', BLACK_KING:'♔', BLACK_PAWN:'♙', EMPTY:'·'}
    for i, row in enumerate(pos.board.split()):
        print(' ', 8-i, ' '.join(uni_pieces.get(p, p) for p in row))
    print('    a b c d e f g h \n\n')


def timeit():
    moves2 = [(84, 64), (85, 65), (97, 76), (97, 76), (92, 73), (92, 73), (93, 66), (96, 63), (76, 55), (76, 64), (66, 55), (73, 54), (86, 76), (54, 46), (82, 73), (84, 74), (85, 75), (95, 51), (87, 77), (51, 84), (96, 52), (86, 76), (52, 74), (83, 73), (74, 56), (74, 63), (55, 66), (82, 62), (66, 57), (73, 62), (94, 74), (84, 83), (95, 97), (94, 96), (91, 94), (96, 97), (96, 95), (93, 82), (88, 78), (91, 94), (81, 71), (88, 78), (71, 61), (81, 71), (61, 51), (71, 61), (74, 85), (82, 73), (85, 74), (83, 72), (74, 85), (73, 82), (85, 74), (72, 74), (74, 85), (82, 73), (85, 74), (74, 85), (74, 85), (85, 86), (85, 86), (86, 68), (86, 84), (68, 38), (84, 74), (38, 37), (74, 56), (95, 75), (56, 47), (75, 74), (47, 74), (61, 51), (77, 68), (62, 51), (94, 92), (94, 92), (97, 98), (37, 38), (92, 42), (92, 42), (74, 56), (74, 75), (95, 94), (97, 88), (94, 92), (88, 98), (42, 32), (38, 56)]
    sdepth = 3
    for run in range(2):
        if run == 0:
            print('Run 1, with jitting:')
        else:
            print('Run', run + 1, 'precompiled:')

        pos = Position(initial, 0, (True,True), (True,True), 0, 0)
        hist = typed.Dict.empty(
            key_type=types.int64,
            value_type=types.int64,
        )
        hist[pos.str] = 1
        searcher = Searcher()

        t1 = time.time()
        mi = 0
        while True:
            if pos.score <= -MATE_LOWER:
                break

            i = 0
            for _, move, score in searcher.search(pos, hist):
                i += 1
                if i == sdepth:
                    break

            if score == MATE_UPPER:
                break
            pos = pos.move(move, pst)
            if sdepth == 2 and run == 0:
                assert move == moves2[mi]
                mi += 1
            print(move)

            hist[pos.str] = 1

        print("took", time.time() - t1)


if __name__ == '__main__':
    timeit()

