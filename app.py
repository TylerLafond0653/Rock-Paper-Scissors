from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
from enum import Enum
from collections import Counter

app = Flask(__name__)

class HandShape(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    UNKNOWN = -1

class GameState:
    def __init__(self):
        self.phase = "IDLE" 
        self.mode = "single"
        self.start_time = 0
        self.judge_timer_start = 0 
        
        self.p1_name = "Player 1"
        self.p2_name = "AI"
        self.p1_score = 0
        self.p2_score = 0
        self.target_wins = 3 
        
        # --- NEW: ADVICE TOGGLE ---
        self.show_advice = True 
        
        self.ai_hands = []
        self.user_hands_snapshot = []
        self.frozen_strategy = "..."
        self.frozen_prediction = "ANALYZING..."
        
        self.last_prediction_raw = None 
        self.ai_stats = {"R": 0, "P": 0, "S": 0, "conf": 0, "pred": "NONE"}
        
        self.stats_history = {
            "total_rounds": 0,
            "ai_correct_guesses": 0,
            "player_moves": [] 
        }
        
        self.winner_msg = ""
        self.winner_color = (255, 255, 255)
        self.game_over_msg = ""
        self.round_result_type = ""
        self.final_report = {}

state = GameState()

class MarkovPredictor:
    def __init__(self):
        self.matrix = {} 
        self.last_move = None
        
    def update(self, current_move):
        if self.last_move is not None:
            transition = (self.last_move, current_move)
            self.matrix[transition] = self.matrix.get(transition, 0) + 1
        self.last_move = current_move
        
    def get_probabilities(self):
        if self.last_move is None: return 33, 33, 33, None
        
        r_count = self.matrix.get((self.last_move, HandShape.ROCK), 0)
        p_count = self.matrix.get((self.last_move, HandShape.PAPER), 0)
        s_count = self.matrix.get((self.last_move, HandShape.SCISSORS), 0)
        total = r_count + p_count + s_count + 0.001 
        
        r_pct = int((r_count / total) * 100)
        p_pct = int((p_count / total) * 100)
        s_pct = int((s_count / total) * 100)
        
        best_hand = None
        if r_count > p_count and r_count > s_count: best_hand = HandShape.ROCK
        elif p_count > r_count and p_count > s_count: best_hand = HandShape.PAPER
        elif s_count > r_count and s_count > p_count: best_hand = HandShape.SCISSORS
        
        return r_pct, p_pct, s_pct, best_hand
    
    def predict_next(self):
        _, _, _, best = self.get_probabilities()
        return best, 0

class GameStrategy:
    def __init__(self):
        self.rules = {
            (HandShape.ROCK, HandShape.SCISSORS): 1, (HandShape.ROCK, HandShape.PAPER): -1, (HandShape.ROCK, HandShape.ROCK): 0,
            (HandShape.PAPER, HandShape.ROCK): 1, (HandShape.PAPER, HandShape.SCISSORS): -1, (HandShape.PAPER, HandShape.PAPER): 0,
            (HandShape.SCISSORS, HandShape.PAPER): 1, (HandShape.SCISSORS, HandShape.ROCK): -1, (HandShape.SCISSORS, HandShape.SCISSORS): 0
        }
    def determine_winner(self, my_hand, opp_hand):
        return self.rules.get((my_hand, opp_hand), 0)

    def solve_nash(self, my_hands, opp_hands, prediction=None):
        if prediction is not None and prediction in opp_hands:
            score0 = self.determine_winner(my_hands[0], prediction)
            score1 = self.determine_winner(my_hands[1], prediction)
            if score0 > score1: return my_hands[0], f"KILLER: {my_hands[0].name}"
            elif score1 > score0: return my_hands[1], f"KILLER: {my_hands[1].name}"

        score0 = self.determine_winner(my_hands[0], opp_hands[0]) + self.determine_winner(my_hands[0], opp_hands[1])
        score1 = self.determine_winner(my_hands[1], opp_hands[0]) + self.determine_winner(my_hands[1], opp_hands[1])
        rec = my_hands[0] if score0 >= score1 else my_hands[1]
        return rec, f"KEEP {rec.name}"

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=4, min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

    def get_dist(self, p1, p2): return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def classify_gesture(self, hand_landmarks):
        thumb = self.get_dist(hand_landmarks.landmark[0], hand_landmarks.landmark[4]) > self.get_dist(hand_landmarks.landmark[0], hand_landmarks.landmark[2])
        index = self.get_dist(hand_landmarks.landmark[0], hand_landmarks.landmark[8]) > self.get_dist(hand_landmarks.landmark[0], hand_landmarks.landmark[6])
        middle = self.get_dist(hand_landmarks.landmark[0], hand_landmarks.landmark[12]) > self.get_dist(hand_landmarks.landmark[0], hand_landmarks.landmark[10])
        ring = self.get_dist(hand_landmarks.landmark[0], hand_landmarks.landmark[16]) > self.get_dist(hand_landmarks.landmark[0], hand_landmarks.landmark[14])
        pinky = self.get_dist(hand_landmarks.landmark[0], hand_landmarks.landmark[20]) > self.get_dist(hand_landmarks.landmark[0], hand_landmarks.landmark[18])
        
        total = sum([thumb, index, middle, ring, pinky])
        if total <= 1: return HandShape.ROCK
        if index and middle and not ring and not pinky: return HandShape.SCISSORS
        if total >= 4: return HandShape.PAPER
        if total == 3 and index and middle: return HandShape.SCISSORS
        return HandShape.UNKNOWN

    def process(self, frame):
        if frame is None: return None, []
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        detected = [] 
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                y_pos = hand_lms.landmark[0].y
                detected.append((self.classify_gesture(hand_lms), y_pos))
                color = (0, 255, 0) if y_pos > 0.5 else (0, 0, 255)
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                                            self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2),
                                            self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))
        return frame, detected

detector = HandDetector()
strategy = GameStrategy()
predictor = MarkovPredictor()

def generate_report():
    history = state.stats_history
    total = history["total_rounds"]
    if total == 0: return
    accuracy = int((history["ai_correct_guesses"] / total) * 100)
    moves = history["player_moves"]
    counts = Counter(moves)
    if not counts: return
    fav_hand, fav_count = counts.most_common(1)[0]
    predictability = int((fav_count / total) * 100)
    state.final_report = {
        "accuracy": f"{accuracy}%",
        "predictability": f"{predictability}%",
        "favorite": fav_hand.name
    }

def perform_judgement(p1_hand, p2_hand):
    state.stats_history["total_rounds"] += 1
    state.stats_history["player_moves"].append(p1_hand)
    if state.last_prediction_raw == p1_hand:
        state.stats_history["ai_correct_guesses"] += 1

    res = strategy.determine_winner(p1_hand, p2_hand)
    
    if res == 1: 
        state.winner_msg = "ROUND WON!"
        state.winner_color = (0, 255, 0)
        state.p1_score += 1
        state.round_result_type = "VICTORY"
    elif res == -1: 
        state.winner_msg = "ROUND LOST!"
        state.winner_color = (0, 0, 255)
        state.p2_score += 1
        state.round_result_type = "DEFEAT"
    else: 
        state.winner_msg = "DRAW"
        state.winner_color = (255, 255, 0)
        state.round_result_type = "DRAW"
    
    predictor.update(p2_hand)

    if state.p1_score >= state.target_wins:
        state.game_over_msg = f"{state.p1_name.upper()} WINS!"
        state.phase = "GAME_OVER"
        generate_report()
    elif state.p2_score >= state.target_wins:
        state.game_over_msg = f"{state.p2_name.upper()} WINS!"
        state.phase = "GAME_OVER"
        generate_report()
    else:
        state.phase = "JUDGE"

def draw_bar(overlay, x, y, w, h, pct, color):
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 255, 255), 1) 
    fill_w = int(w * (pct / 100.0))
    if fill_w > 0:
        cv2.rectangle(overlay, (x, y), (x+fill_w, y+h), color, -1)

def draw_hud(frame, opp_display, my_display, strategy_text, center_msg, msg_color):
    if frame is None: return None
    h, w, _ = frame.shape
    
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"{state.p1_name}: {state.p1_score}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    p2_text = f"{state.p2_name}: {state.p2_score}"
    ts = cv2.getTextSize(p2_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.putText(frame, p2_text, (w - ts[0] - 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    sidebar_w = int(w * 0.22)
    if sidebar_w < 220: sidebar_w = 220
    sidebar_x = w - sidebar_w
    cv2.rectangle(frame, (sidebar_x, 60), (w, 320), (0, 0, 0), -1)
    
    cv2.putText(frame, "THE GUESSER", (sidebar_x + 10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.line(frame, (sidebar_x + 10, 95), (w-10, 95), (255, 255, 0), 1)
    
    stats = state.ai_stats
    bar_max_w = sidebar_w - 90 
    bar_h = 10
    
    cv2.putText(frame, "R", (sidebar_x + 10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    draw_bar(frame, sidebar_x + 70, 110, bar_max_w, bar_h, stats["R"], (0, 0, 255))
    cv2.putText(frame, "P", (sidebar_x + 10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    draw_bar(frame, sidebar_x + 70, 130, bar_max_w, bar_h, stats["P"], (0, 255, 0))
    cv2.putText(frame, "S", (sidebar_x + 10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    draw_bar(frame, sidebar_x + 70, 150, bar_max_w, bar_h, stats["S"], (0, 255, 255))
    
    cv2.putText(frame, "PREDICTION:", (sidebar_x + 10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    pred_txt = stats["pred"] if stats["pred"] != "NONE" else "WAITING..."
    cv2.putText(frame, pred_txt, (sidebar_x + 10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.putText(frame, "COUNTER:", (sidebar_x + 10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    ai_move_text = "..."
    ai_color = (100, 100, 100)
    if state.mode == 'single' and len(state.ai_hands) > 0 and state.phase != "IDLE":
        if len(state.ai_hands) == 1: 
            ai_move_text = state.ai_hands[0].name
            ai_color = (0, 0, 255)
        elif len(state.ai_hands) == 2: 
            ai_move_text = f"{state.ai_hands[0].name} + {state.ai_hands[1].name}"
            ai_color = (0, 165, 255)
            
    font_s = 0.6 if len(ai_move_text) < 12 else 0.4
    cv2.putText(frame, ai_move_text, (sidebar_x + 10, 285), cv2.FONT_HERSHEY_SIMPLEX, font_s, ai_color, 2)

    cv2.rectangle(frame, (0, h-50), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "YOU:", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    my_str = " + ".join([h.name for h in my_display]) if my_display else "..."
    cv2.putText(frame, my_str, (70, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # --- CENTER ADVICE LOGIC (THE TOGGLE) ---
    advice_text = strategy_text
    advice_color = (255, 255, 0)
    
    # If toggle is OFF, hide the text
    if not state.show_advice:
        advice_text = "DATA ENCRYPTED"
        advice_color = (100, 100, 100)

    advice_size = cv2.getTextSize(advice_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    advice_x = (w - advice_size[0]) // 2
    cv2.putText(frame, advice_text, (advice_x, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, advice_color, 2)

    if center_msg:
        if state.phase == "GAME_OVER":
            cv2.rectangle(frame, (0, h//2 - 100), (w, h//2 + 100), (0,0,0), -1)
            cv2.putText(frame, center_msg, (w//2 - 300, h//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, msg_color, 4)
        else:
            text_size = cv2.getTextSize(center_msg, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(frame, center_msg, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 6)
            cv2.putText(frame, center_msg, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, msg_color, 3)

    return frame

def generate_frames():
    global state
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened(): return 

    while True:
        success, frame = cap.read()
        if not success or frame is None: continue 
        frame = cv2.flip(frame, 1)
        frame, raw_data = detector.process(frame)
        if frame is None: continue 

        all_bottom = [d[0] for d in raw_data if d[1] > 0.5]
        all_top = [d[0] for d in raw_data if d[1] <= 0.5]

        center_msg, msg_color = "", (255, 255, 255)
        current_opp, current_my = [], []
        
        r_pct, p_pct, s_pct, best_guess = predictor.get_probabilities()
        state.ai_stats["R"] = r_pct
        state.ai_stats["P"] = p_pct
        state.ai_stats["S"] = s_pct
        state.ai_stats["pred"] = best_guess.name if best_guess else "NONE"

        if state.phase == "IDLE":
            center_msg, msg_color = "PRESS INITIALIZE", (200, 200, 200)
            current_my = all_bottom[:2]
            
        elif state.phase == "COUNTDOWN":
            elapsed = time.time() - state.start_time
            countdown = 3 - int(elapsed)
            if countdown <= 0:
                state.phase = "SHOOT"
                state.start_time = time.time()
                if state.mode == 'single':
                    state.ai_hands = random.choices([HandShape.ROCK, HandShape.PAPER, HandShape.SCISSORS], k=2)
            else:
                center_msg, msg_color = str(countdown), (0, 255, 255)

        elif state.phase == "SHOOT":
            center_msg, msg_color = "SHOW HANDS!", (0, 0, 255)
            if state.mode == 'single':
                current_opp = state.ai_hands
                current_my = all_bottom[:2]
            else:
                current_opp = all_top[:2]
                current_my = all_bottom[:2]

            if len(current_my) >= 2 and len(current_opp) >= 2:
                state.user_hands_snapshot = current_my
                state.last_prediction_raw = best_guess
                state.frozen_strategy = strategy.solve_nash(current_my, current_opp, best_guess)[1]
                
                if time.time() - state.start_time > 0.5:
                    state.phase = "DECIDE"
                    state.judge_timer_start = 0

        elif state.phase == "DECIDE":
            # Advice is determined in draw_hud based on toggle state
            current_opp = state.ai_hands if state.mode == 'single' else all_top[:2]
            current_my = all_bottom

            ready_to_judge = False
            if state.mode == 'single':
                if len(current_my) == 1: ready_to_judge = True
            else:
                if len(current_my) == 1 and len(current_opp) == 1: ready_to_judge = True
            
            if ready_to_judge:
                if state.judge_timer_start == 0:
                    state.judge_timer_start = time.time()
                elif time.time() - state.judge_timer_start > 0.8: 
                    user_h = current_my[0]
                    opp_h = current_opp[0] if state.mode == 'multi' else state.ai_hands[0]
                    perform_judgement(user_h, opp_h)
            else:
                state.judge_timer_start = 0

        elif state.phase == "JUDGE":
            center_msg, msg_color = state.winner_msg, state.winner_color
            if len(state.user_hands_snapshot) > 0: current_my = [state.user_hands_snapshot[0]]
            if state.mode == 'single' and len(state.ai_hands) > 0: current_opp = [state.ai_hands[0]]

        elif state.phase == "GAME_OVER":
            center_msg, msg_color = state.game_over_msg, (0, 255, 255)

        frame = draw_hud(frame, current_opp, current_my, state.frozen_strategy, center_msg, msg_color)
        
        if frame is None: continue 

        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret: continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception: continue

    cap.release()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/action/setup_game')
def setup_game():
    state.p1_name = request.args.get('p1', 'Player 1')
    state.p2_name = request.args.get('p2', 'AI')
    state.target_wins = int(request.args.get('limit', 3))
    state.p1_score = 0
    state.p2_score = 0
    state.phase = "IDLE"
    global predictor
    predictor = MarkovPredictor()
    return jsonify(status="ok")

@app.route('/play/<mode>')
def play(mode):
    state.mode = mode
    state.phase = "IDLE"
    return render_template('game.html', mode=mode)

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/action/start_round')
def start_round():
    if state.phase == "GAME_OVER": return jsonify(status="game_over")
    
    # --- GET TOGGLE STATE ---
    assist = request.args.get('assist', 'true') == 'true'
    state.show_advice = assist
    
    state.phase = "COUNTDOWN"
    state.start_time = time.time()
    return jsonify(status="ok")

@app.route('/action/reset')
def reset():
    if state.phase == "GAME_OVER": return jsonify(status="game_over")
    state.phase = "IDLE"
    return jsonify(status="ok")

@app.route('/status')
def get_status():
    return jsonify(
        phase=state.phase, 
        winner=state.winner_msg,
        game_winner=state.game_over_msg,
        result_type=state.round_result_type,
        p1_name=state.p1_name if state.p1_score >= state.target_wins else state.p2_name,
        report=state.final_report 
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)