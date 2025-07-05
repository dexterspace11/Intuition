# -------------------- Intuitive Neural AI - Streamlit Dashboard (Auto-Updating) --------------------
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pickle
import networkx as nx

# ---------------- Memory Structures -------------------
class EpisodicMemory:
    def __init__(self):
        self.episodes = {}
        self.current_episode = None

    def create_episode(self, timestamp):
        self.current_episode = timestamp
        self.episodes[timestamp] = {'patterns': [], 'emotional_tags': []}

    def store_pattern(self, pattern, emotional_tag):
        if self.current_episode is None:
            self.create_episode(datetime.now())
        self.episodes[self.current_episode]['patterns'].append(pattern)
        self.episodes[self.current_episode]['emotional_tags'].append(emotional_tag)

class WorkingMemory:
    def __init__(self, capacity=20):
        self.capacity = capacity
        self.patterns = []

    def store(self, pattern):
        if len(self.patterns) >= self.capacity:
            self.patterns.pop(0)
        self.patterns.append(pattern)

    def replay(self):
        return list(self.patterns)

# ---------------- Curiosity Engine -------------------
class CuriosityEngine:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def evaluate_novelty(self, similarity):
        return 1.0 - similarity

    def should_grow(self, similarity):
        return self.evaluate_novelty(similarity) > self.threshold

# ---------------- Pattern Recognizer -------------------
class PatternRecognizer:
    def __init__(self):
        self.pattern_memory = {}

    def detect_patterns(self, sequence):
        patterns = []
        for length in range(2, len(sequence)):
            for i in range(len(sequence)-length+1):
                window = tuple(sequence[i:i+length])
                if window in self.pattern_memory:
                    patterns.append((window, self.pattern_memory[window]))
        return sorted(patterns, key=lambda x: x[1], reverse=True)

    def reinforce(self, pattern):
        self.pattern_memory[pattern] = self.pattern_memory.get(pattern, 0.0) + 0.1

    def predict_next(self, history):
        detected = self.detect_patterns(history)
        if detected:
            return detected[0][0][-1]
        return random.randint(1, 9)

# ---------------- Neural Structures -------------------
class HybridNeuralUnit:
    def __init__(self, position):
        self.position = np.array(position)
        self.age = 0
        self.usage_count = 0
        self.emotional_weight = 1.0
        self.connections = []

    def quantum_similarity(self, input_pattern):
        diff = np.abs(np.array(input_pattern) - self.position)
        dist = np.sqrt(np.sum(diff ** 2))
        return np.exp(-dist) * np.exp(-self.age / 100.0)

    def update(self, reward):
        self.usage_count += 1
        self.age = 0
        self.emotional_weight = 1.0 + reward

class IntuitiveNeuralNetwork:
    def __init__(self):
        self.units = []
        self.memory = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.recognizer = PatternRecognizer()
        self.curiosity = CuriosityEngine()

    def grow(self, input_pattern):
        new_unit = HybridNeuralUnit(input_pattern)
        for unit in self.units:
            if random.random() < 0.3:
                unit.connections.append(len(self.units))
                new_unit.connections.append(self.units.index(unit))
        self.units.append(new_unit)
        return new_unit

    def process_input(self, input_pattern, reward=0):
        if not self.units:
            return self.grow(input_pattern), 0.0

        similarities = [(u, u.quantum_similarity(input_pattern)) for u in self.units]
        best_unit, best_similarity = max(similarities, key=lambda x: x[1])
        best_unit.update(reward)

        emotional_tag = 1.0 + best_similarity
        self.memory.store(input_pattern)
        self.episodic.store_pattern(input_pattern, emotional_tag)

        if self.curiosity.should_grow(best_similarity):
            return self.grow(input_pattern), 0.0
        return best_unit, best_similarity

    def predict_next_sequence(self, history_flat):
        pred = []
        for _ in range(3):
            p = self.recognizer.predict_next(history_flat)
            pred.append(p)
            history_flat.append(p)
        return pred

    def train_on_sequence(self, sequence, reward=0):
        flat_history = [n for seq in st.session_state.history for n in seq][-30:]
        self.recognizer.reinforce(tuple(sequence))
        _, sim = self.process_input(sequence, reward)
        return sim

    def dream(self):
        for dream in self.memory.replay():
            self.train_on_sequence(dream, reward=0.05)

# ---------------- Helper Functions -------------------
def generate_lottery_sequence():
    return [random.SystemRandom().randint(1, 9) for _ in range(3)]

def plot_accuracy(acc):
    fig, ax = plt.subplots()
    ax.plot(acc, label='Intuition Accuracy')
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy Over Time")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Accuracy")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

def draw_neural_graph(network):
    G = nx.Graph()
    for i, unit in enumerate(network.units):
        G.add_node(i)
        for j in unit.connections:
            G.add_edge(i, j)
    fig, ax = plt.subplots(figsize=(8, 5))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, ax=ax)
    st.pyplot(fig)

# ---------------- Persistence -------------------
def save_state():
    with open("net_state.pkl", "wb") as f:
        pickle.dump({
            'net': st.session_state.net,
            'history': st.session_state.history,
            'predictions': st.session_state.predictions,
            'accuracies': st.session_state.accuracies,
            'round_num': st.session_state.round_num,
            'last_run': st.session_state.last_run
        }, f)

def load_state():
    if os.path.exists("net_state.pkl"):
        with open("net_state.pkl", "rb") as f:
            state = pickle.load(f)
            st.session_state.net = state['net']
            st.session_state.history = state['history']
            st.session_state.predictions = state['predictions']
            st.session_state.accuracies = state['accuracies']
            st.session_state.round_num = state['round_num']
            st.session_state.last_run = state['last_run']

# ---------------- Streamlit UI -------------------
st.set_page_config(page_title="Intuitive Neural AI", layout="wide")
st_autorefresh(interval=60000, limit=None, key="auto_refresh")
st.title("🤖 Intuitive Neural AI - Self Growing, Pattern-Aware Predictor")

if 'net' not in st.session_state:
    load_state()
    if 'net' not in st.session_state:
        st.session_state.net = IntuitiveNeuralNetwork()
        st.session_state.history = []
        st.session_state.predictions = []
        st.session_state.accuracies = []
        st.session_state.round_num = 0
        st.session_state.last_run = time.time()

interval_seconds = 60
now = time.time()
if now - st.session_state.last_run >= interval_seconds:
    st.session_state.last_run = now
    net = st.session_state.net
    history = st.session_state.history
    predictions = st.session_state.predictions
    accuracies = st.session_state.accuracies
    round_num = st.session_state.round_num + 1

    actual = generate_lottery_sequence()
    flat_history = [n for seq in history for n in seq][-30:]
    prediction = net.predict_next_sequence(flat_history.copy())
    predictions.append(prediction)
    history.append(actual)

    correct = sum([1 for a, p in zip(actual, prediction) if a == p])
    accuracy = correct / 3.0
    accuracies.append(accuracy)
    sim_score = net.train_on_sequence(actual, reward=accuracy)

    if round_num % 5 == 0:
        net.dream()

    st.session_state.history = history
    st.session_state.predictions = predictions
    st.session_state.accuracies = accuracies
    st.session_state.round_num = round_num

    save_state()

st.subheader(f"Round {st.session_state.round_num}")
st.markdown(f"⏳ Updated at: {datetime.now().strftime('%H:%M:%S')}")

if st.session_state.round_num > 0:
    st.write(f"Actual: {st.session_state.history[-1]}, Predicted: {st.session_state.predictions[-1]}, Match: {st.session_state.accuracies[-1]:.2f}")
    plot_accuracy(st.session_state.accuracies)

    st.subheader("🧠 Neural Units")
    df_units = pd.DataFrame([
        {
            'Age': u.age,
            'Usage Count': u.usage_count,
            'Emotional Weight': round(u.emotional_weight, 2),
            'Position': list(u.position),
            'Connections': len(u.connections)
        }
        for u in st.session_state.net.units
    ])
    st.dataframe(df_units)

    st.subheader("🔗 Neural Network Connections")
    draw_neural_graph(st.session_state.net)

    st.subheader("🧩 Working Memory Snapshots")
    st.write(st.session_state.net.memory.replay())

    st.subheader("🧠 Episodic Memory (Latest)")
    if st.session_state.net.episodic.episodes:
        latest = list(st.session_state.net.episodic.episodes.items())[-1]
        st.write(latest)

st.caption("Auto-updating every 60 seconds to grow and learn continuously. Memory is saved and restored between runs.")