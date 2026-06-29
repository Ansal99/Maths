import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import re

class HintEngine:
    def __init__(self, csv_path='data/maths_only.csv'):
        self.csv_path = csv_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.df = None
        self.embeddings = None
        self._load_data()

    def _load_data(self):
        self.df = pd.read_csv(self.csv_path)

        print(f"📊 Total rows in CSV: {len(self.df)}")
        print(f"📋 Columns: {self.df.columns.tolist()}")

        # Step 1: Math context filter
        if 'context' in self.df.columns:
            self.df = self.df[
                self.df['context'].str.contains('Mathematics|Maths|Math', case=False, na=False)
            ].reset_index(drop=True)

        print(f"📌 After context filter: {len(self.df)} questions")

        # Step 2: Non-math questions hatao
        math_keywords = [
            'equation', 'formula', 'calculate', 'number', 'triangle', 'circle',
            'square', 'rectangle', 'angle', 'area', 'perimeter', 'volume',
            'integer', 'fraction', 'decimal', 'percentage', 'ratio', 'proportion',
            'algebra', 'geometry', 'arithmetic', 'probability', 'statistics',
            'factor', 'multiple', 'prime', 'digit', 'variable', 'expression',
            'polynomial', 'theorem', 'proof', 'graph', 'coordinate', 'matrix',
            'addition', 'subtraction', 'multiplication', 'division', 'sum',
            'product', 'difference', 'quotient', 'remainder', 'solve', 'find',
            'value', 'LHS', 'RHS', 'property', 'commutative', 'associative',
            'distributive', 'whole number', 'natural number', 'rational',
            'irrational', 'real number', 'profit', 'loss', 'interest',
            'principal', 'rate', 'speed', 'distance', 'parallel', 'perpendicular',
            'quadrilateral', 'polygon', 'cylinder', 'cone', 'sphere', 'cube',
            'cuboid', 'symmetry', 'rotation', 'mensuration', 'mean', 'median',
            'mode', 'range', 'quadratic', 'linear', 'progression', 'sequence',
            'series', 'matrix', 'determinant', 'trigonometry', 'sine', 'cosine',
            'tangent', 'logarithm', 'exponent', 'power', 'root', 'set',
            'subset', 'union', 'intersection', 'complement', 'function',
            'domain', 'range', 'slope', 'intercept', 'parabola', 'hyperbola',
            'ellipse', 'circle', 'radius', 'diameter', 'chord', 'arc',
            'sector', 'segment', 'height', 'base', 'hypotenuse', 'adjacent',
            'opposite', 'degree', 'radian', 'pi', 'infinity', 'limit',
            'derivative', 'integral', 'vector', 'scalar', 'magnitude',
            'direction', 'resultant', 'component', 'projection', 'dot product',
            'cross product', 'matrix', 'inverse', 'transpose', 'eigenvalue',
            'eigenvector', 'probability', 'permutation', 'combination',
            'factorial', 'binomial', 'coefficient', 'polynomial', 'degree',
            'leading coefficient', 'constant term', 'monomial', 'binomial',
            'trinomial', 'like terms', 'unlike terms', 'simplify', 'expand',
            'factorise', 'factorize', 'HCF', 'LCM', 'GCD', 'prime factorization',
            'divisibility', 'even', 'odd', 'positive', 'negative', 'zero',
            'absolute value', 'modulus', 'inequality', 'greater than', 'less than',
            'equal', 'not equal', 'approximately', 'estimate', 'round', 'truncate',
            'significant figures', 'scientific notation', 'standard form',
            'expanded form', 'place value', 'face value', 'numeral', 'digit',
            'units', 'tens', 'hundreds', 'thousands', 'lakhs', 'crores',
            'unitary method', 'direct proportion', 'inverse proportion',
            'compound interest', 'simple interest', 'discount', 'tax', 'VAT',
            'profit percent', 'loss percent', 'cost price', 'selling price',
            'marked price', 'commission', 'partnership', 'work', 'time',
            'pipe', 'cistern', 'train', 'boat', 'stream', 'upstream',
            'downstream', 'average', 'weighted average', 'frequency',
            'class interval', 'histogram', 'bar graph', 'pie chart',
            'line graph', 'scatter plot', 'correlation', 'regression',
            'standard deviation', 'variance', 'quartile', 'percentile',
            'sample', 'population', 'random', 'event', 'outcome', 'experiment',
            'trial', 'sample space', 'favorable outcome', 'equally likely',
            'mutually exclusive', 'independent', 'dependent', 'conditional',
            'Bayes', 'normal distribution', 'binomial distribution'
        ]

        pattern = '|'.join(math_keywords)
        self.df = self.df[
            self.df['question'].str.contains(pattern, case=False, na=False)
        ].reset_index(drop=True)

        print(f"✅ Clean Maths questions loaded: {len(self.df)}")

        # Embeddings banao ya cache se load karo
        cache_file = 'data/embeddings_cache.pkl'
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            print("✅ Embeddings loaded from cache")
        else:
            print("⏳ Generating embeddings... (first time only)")
            questions = self.df['question'].fillna('').tolist()
            self.embeddings = self.model.encode(questions, show_progress_bar=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            print("✅ Embeddings cached!")

    def get_hint(self, user_question, top_k=3):
        query_embedding = self.model.encode([user_question])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        best_score = similarities[top_indices[0]]

        if best_score < 0.45:
            return {
                "found": False,
                "message": "❌ Yeh Math ka question nahi lagta, ya dataset mein nahi hai. Koi aur Math question puchho!",
                "hints": [],
                "score": float(best_score)
            }

        best_match = self.df.iloc[top_indices[0]]
        hints = self._generate_progressive_hints(best_match)

        return {
            "found": True,
            "matched_question": best_match.get('question', ''),
            "hints": hints,
            "score": float(best_score),
            "topic": best_match.get('context', 'Mathematics')
        }

    def _generate_progressive_hints(self, row):
        raw_hint = str(row.get('hint', row.get('answer', '')))

        if not raw_hint or raw_hint == 'nan':
            return ["Is question ko solve karne ke liye pehle basic formula yaad karo."]

        sentences = re.split(r'(?<=[.!?])\s+', raw_hint.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        if len(sentences) == 0:
            return [raw_hint]
        elif len(sentences) == 1:
            return [f"💡 Hint 1: {sentences[0]}"]
        else:
            hints = []
            for i, sentence in enumerate(sentences[:3]):
                if i == 0:
                    hints.append(f"💡 Hint 1: {sentence}")
                elif i == 1:
                    hints.append(f"🔍 Hint 2: {sentence}")
                else:
                    hints.append(f"🧩 Hint 3 (Last): {sentence}")
            return hints