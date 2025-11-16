"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')

class AdvancedThompsonSamplingBandit:
    def __init__(self):
        self.models = {}
        self.feature_columns = None
        self.actions = ['Mens E-Mail', 'Womens E-Mail', 'No E-Mail']
        self.scaler = StandardScaler()
        self.best_strategy = 'thompson_sampling'
        self.action_encoders = {}
        self.propensity_model = None
        self.reward_models = {}
        self.thompson_samples = 100  
        
    def prepare_features(self, df, is_train=True):
        """Подготовка признаков"""
        df = df.copy()
        
        numeric_cols = []
        if 'recency' in df.columns:
            numeric_cols.append('recency')
            df['recency_log'] = np.log1p(df['recency'])
            df['recency_sqrt'] = np.sqrt(df['recency'])
            df['recency_inv'] = 1 / (df['recency'] + 1)
            numeric_cols.extend(['recency_log', 'recency_sqrt', 'recency_inv'])
            
        if 'history' in df.columns:
            numeric_cols.append('history')
            df['history_log'] = np.log1p(df['history'])
            df['history_sqrt'] = np.sqrt(df['history'])
            df['history_boxcox'] = np.log1p(df['history'] + 1)  
            numeric_cols.extend(['history_log', 'history_sqrt', 'history_boxcox'])
            
            df['history_segment_fine'] = pd.cut(df['history'], 
                                              bins=[0, 10, 25, 50, 100, 200, 500, np.inf], 
                                              labels=False)
            numeric_cols.append('history_segment_fine')
        
        binary_cols = []
        for col in ['mens', 'womens', 'newbie']:
            if col in df.columns:
                binary_cols.append(col)
        
        if 'mens' in df.columns and 'womens' in df.columns:
            df['any_interest'] = (df['mens'] == 1) | (df['womens'] == 1)
            df['both_interests'] = (df['mens'] == 1) & (df['womens'] == 1)
            df['interest_strength'] = df['mens'] + df['womens']
            df['interest_balance'] = np.abs(df['mens'] - df['womens'])
            binary_cols.extend(['any_interest', 'both_interests'])
            numeric_cols.extend(['interest_strength', 'interest_balance'])
        
        if 'recency' in df.columns and 'history' in df.columns:
            df['recency_history_ratio'] = df['history'] / (df['recency'] + 1)
            df['customer_activity'] = df['history'] * np.exp(-df['recency'] / 12)
            df['value_recency_interaction'] = df['history_log'] * df['recency_inv']
            numeric_cols.extend(['recency_history_ratio', 'customer_activity', 'value_recency_interaction'])
        
        categorical_cols = []
        for col in ['zip_code', 'channel']:
            if col in df.columns:
                df[col] = df[col].fillna('missing')
                
                if is_train:
                    global_mean = df['visit'].mean() if 'visit' in df.columns else 0.1
                    stats = df.groupby(col).agg({
                        'visit': ['count', 'mean'] if 'visit' in df.columns else ('dummy', ['count', lambda x: global_mean])
                    }).round(4)
                    
                    stats.columns = ['count', 'mean']
                    stats['smooth_mean'] = (stats['count'] * stats['mean'] + 10 * global_mean) / (stats['count'] + 10)
                    
                    self.action_encoders[f'{col}_smooth'] = stats['smooth_mean'].to_dict()
                    self.action_encoders[f'{col}_global_mean'] = global_mean
                
                smooth_encoder = self.action_encoders.get(f'{col}_smooth', {})
                global_mean = self.action_encoders.get(f'{col}_global_mean', 0.1)
                df[f'{col}_smooth'] = df[col].map(smooth_encoder).fillna(global_mean)
                numeric_cols.append(f'{col}_smooth')
                
                if is_train:
                    freq_encoder = df[col].value_counts().to_dict()
                    self.action_encoders[f'{col}_freq'] = freq_encoder
                
                freq_encoder = self.action_encoders.get(f'{col}_freq', {})
                df[f'{col}_freq'] = df[col].map(freq_encoder).fillna(1)
                numeric_cols.append(f'{col}_freq')
        
        if 'history_log' in df.columns and 'recency_log' in df.columns:
            df['history_recency_poly'] = df['history_log'] * df['recency_log']
            df['history_sq'] = df['history_log'] ** 2
            df['recency_sq'] = df['recency_log'] ** 2
            numeric_cols.extend(['history_recency_poly', 'history_sq', 'recency_sq'])
        
        feature_cols = numeric_cols + binary_cols + categorical_cols
        feature_cols = list(dict.fromkeys(feature_cols))
        
        X = df[feature_cols].copy()
        
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in ['float64', 'int64']:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0], inplace=True)
        
        numeric_cols_to_scale = [col for col in numeric_cols if col in X.columns]
        if numeric_cols_to_scale:
            if is_train:
                X[numeric_cols_to_scale] = self.scaler.fit_transform(X[numeric_cols_to_scale])
            else:
                X[numeric_cols_to_scale] = self.scaler.transform(X[numeric_cols_to_scale])
        
        return X, feature_cols
    
    def fit_bayesian_models(self, X_train, y_train, actions_train):
        """Байесовские модели с апостериорным распределением"""
        print("Обучение байесовских моделей...")
        
        for action in self.actions:
            mask = actions_train == action
            action_count = mask.sum()
            
            if action_count > 20:
                try:
                    models = [
                        ('rf', RandomForestClassifier(
                            n_estimators=50,
                            max_depth=8,
                            min_samples_split=15,
                            min_samples_leaf=5,
                            random_state=42
                        )),
                        ('gbm', GradientBoostingClassifier(
                            n_estimators=50,
                            max_depth=5,
                            learning_rate=0.1,
                            random_state=42
                        )),
                        ('xgb', XGBClassifier(
                            n_estimators=50,
                            max_depth=5,
                            learning_rate=0.1,
                            random_state=42,
                            verbosity=0
                        ))
                    ]
                    
                    ensemble = VotingClassifier(estimators=models, voting='soft')
                    ensemble.fit(X_train[mask], y_train[mask])
                    
                    calibrated_model = CalibratedClassifierCV(ensemble, method='isotonic', cv=min(3, action_count//5))
                    calibrated_model.fit(X_train[mask], y_train[mask])
                    
                    self.models[f"bayesian_{action}"] = calibrated_model
                    print(f"  Байесовская модель для {action} обучена")
                    
                except Exception as e:
                    print(f"  Ошибка байесовской модели {action}: {e}")
                    try:
                        simple_model = LogisticRegression(random_state=42)
                        simple_model.fit(X_train[mask], y_train[mask])
                        self.models[f"bayesian_{action}"] = simple_model
                    except:
                        pass
    
    def fit_thompson_sampling(self, X_train, y_train, actions_train):
        """Реализация семплирования Томпсона"""
        print("Настройка семплирования Томпсона...")
        
        for action in self.actions:
            mask = actions_train == action
            if mask.sum() > 10:
                successes = y_train[mask].sum()
                failures = mask.sum() - successes
                
                self.reward_models[f"beta_{action}"] = {
                    'alpha': successes + 1, 
                    'beta': failures + 1,
                    'count': mask.sum(),
                    'empirical_mean': y_train[mask].mean() if mask.sum() > 0 else 0.1
                }
                print(f"  Бета-распределение для {action}: alpha={successes+1}, beta={failures+1}")
    
    def fit_contextual_thompson(self, X_train, y_train, actions_train):
        """Контекстуальное семплирование Томпсона"""
        print("Обучение контекстуального Томпсона...")
        
        for action in self.actions:
            mask = actions_train == action
            if mask.sum() > 30:
                try:
                    success_model = XGBRegressor(
                        n_estimators=50,
                        max_depth=4,
                        learning_rate=0.1,
                        random_state=42,
                        verbosity=0
                    )
                    
                    from sklearn.neighbors import NearestNeighbors
                    
                    prob_model = XGBClassifier(
                        n_estimators=50,
                        max_depth=4,
                        learning_rate=0.1,
                        random_state=42,
                        verbosity=0
                    )
                    
                    prob_model.fit(X_train[mask], y_train[mask])
                    self.models[f"contextual_thompson_{action}"] = prob_model
                    print(f"  Контекстуальная модель для {action} обучена")
                    
                except Exception as e:
                    print(f"  Ошибка контекстуального Томпсона {action}: {e}")
    
    def predict_thompson_samples(self, X, n_samples=100):
        """Семплирование Томпсона для предсказания"""
        print("Применение семплирования Томпсона...")
        
        action_samples = {action: [] for action in self.actions}
        
        for i in range(len(X)):
            sample_rewards = {}
            
            for action in self.actions:
                if f"beta_{action}" in self.reward_models:
                    params = self.reward_models[f"beta_{action}"]
                    samples = np.random.beta(params['alpha'], params['beta'], n_samples)
                    sample_rewards[action] = np.mean(samples)
                else:
                    if f"contextual_thompson_{action}" in self.models:
                        model = self.models[f"contextual_thompson_{action}"]
                        if hasattr(model, 'predict_proba'):
                            prob = model.predict_proba(X.iloc[i:i+1])[:, 1][0]
                            effective_n = 10  
                            alpha = prob * effective_n + 1
                            beta = (1 - prob) * effective_n + 1
                            samples = np.random.beta(alpha, beta, n_samples)
                            sample_rewards[action] = np.mean(samples)
                    else:
                        if f"bayesian_{action}" in self.models:
                            model = self.models[f"bayesian_{action}"]
                            if hasattr(model, 'predict_proba'):
                                prob = model.predict_proba(X.iloc[i:i+1])[:, 1][0]
                                sample_rewards[action] = prob
                        else:
                            sample_rewards[action] = 0.1
            
            for action in self.actions:
                action_samples[action].append(sample_rewards.get(action, 0.1))
        
        for action in self.actions:
            action_samples[action] = np.array(action_samples[action])
        
        return action_samples
    
    def create_thompson_policy(self, action_samples, method='direct'):
        """Создание политики на основе семплирования Томпсона"""
        print("Создание политики Томпсона...")
        
        n_samples = self.thompson_samples
        n_observations = len(next(iter(action_samples.values())))
        
        if method == 'direct':
            best_actions = []
            for i in range(n_observations):
                action_rewards = {}
                for action in self.actions:
                    action_rewards[action] = action_samples[action][i]
                
                best_action = max(action_rewards, key=action_rewards.get)
                best_actions.append(best_action)
            
            probabilities = np.zeros((n_observations, len(self.actions)))
            for i, best_action in enumerate(best_actions):
                probabilities[i, self.actions.index(best_action)] = 1.0
            
        elif method == 'softmax':
            values_matrix = np.column_stack([action_samples[action] for action in self.actions])
            
            values_range = values_matrix.max(axis=1) - values_matrix.min(axis=1)
            temperature = 0.1 / (1 + values_range)
            temperature = np.clip(temperature, 0.05, 0.3)
            
            exp_values = np.zeros_like(values_matrix)
            for i in range(n_observations):
                exp_values[i] = np.exp(values_matrix[i] / temperature[i])
            
            probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
        
        elif method == 'ucb_thompson':
            values_matrix = np.column_stack([action_samples[action] for action in self.actions])
            
            uncertainty_bonus = np.std(values_matrix, axis=1, keepdims=True) * 0.5
            values_with_bonus = values_matrix + uncertainty_bonus
            
            exp_values = np.exp(values_with_bonus * 10)  
            probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
        
        min_prob = 0.01
        probabilities = np.maximum(probabilities, min_prob)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        return probabilities
    
    def fit_ucb_improved(self, X_train, y_train, actions_train):
        """UCB алгоритм"""
        print("Обучение UCB...")
        
        action_stats = {}
        for action in self.actions:
            mask = actions_train == action
            if mask.sum() > 0:
                action_stats[action] = {
                    'mean_reward': y_train[mask].mean(),
                    'count': mask.sum(),
                    'std_reward': y_train[mask].std() if mask.sum() > 1 else 0.1
                }
            else:
                action_stats[action] = {
                    'mean_reward': 0.1,
                    'count': 0,
                    'std_reward': 0.1
                }
        
        self.action_stats = action_stats
    
    def predict_ucb(self, X, t=1000):
        """Предсказание UCB с контекстом"""
        n_observations = len(X)
        ucb_scores = {action: np.zeros(n_observations) for action in self.actions}
        
        for action in self.actions:
            stats = self.action_stats.get(action, {'mean_reward': 0.1, 'count': 1, 'std_reward': 0.1})
            
            if stats['count'] > 0:
                exploration_bonus = np.sqrt(2 * np.log(t) / stats['count'])
            else:
                exploration_bonus = np.sqrt(2 * np.log(t))
            
            if f"bayesian_{action}" in self.models:
                model = self.models[f"bayesian_{action}"]
                if hasattr(model, 'predict_proba'):
                    contextual_mean = model.predict_proba(X)[:, 1]
                    weight = min(stats['count'] / 100, 1.0) 
                    combined_mean = weight * stats['mean_reward'] + (1 - weight) * contextual_mean
                else:
                    combined_mean = stats['mean_reward']
            else:
                combined_mean = stats['mean_reward']
            
            ucb_scores[action] = combined_mean + exploration_bonus
        
        ucb_matrix = np.column_stack([ucb_scores[action] for action in self.actions])
        exp_scores = np.exp(ucb_matrix * 5)  
        probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        
        return probabilities
    
    def fit(self, train_data):
        print("Подготовка данных...")
        X, self.feature_columns = self.prepare_features(train_data, is_train=True)
        y = train_data['visit'].values
        actions = train_data['segment'].values
        
        print(f"Используется {len(self.feature_columns)} признаков")
        
        X_train, X_val, y_train, y_val, actions_train, actions_val = train_test_split(
            X, y, actions, test_size=0.2, random_state=42, stratify=actions
        )
        
        print("\n=== Обучение RL методов ===")
        
        self.fit_bayesian_models(X_train, y_train, actions_train)
        
        self.fit_thompson_sampling(X_train, y_train, actions_train)
        self.fit_contextual_thompson(X_train, y_train, actions_train)
        
        self.fit_ucb_improved(X_train, y_train, actions_train)
        
        self._optimize_policy(X_val, y_val, actions_val)
        
        return self
    
    def _optimize_policy(self, X_val, y_val, actions_val):
        """Оптимизация RL политик"""
        print("\nОптимизация RL политик...")
        
        strategies = {
            'thompson_direct': ('thompson', 'direct'),
            'thompson_softmax': ('thompson', 'softmax'), 
            'thompson_ucb': ('thompson', 'ucb_thompson'),
            'ucb_contextual': ('ucb', None),
            'bayesian_softmax': ('bayesian', 'softmax')
        }
        
        best_score = -np.inf
        best_strategy = None
        best_probs = None
        
        for strategy_name, (method, subtype) in strategies.items():
            try:
                if method == 'thompson':
                    action_samples = self.predict_thompson_samples(X_val)
                    policy_probs = self.create_thompson_policy(action_samples, subtype)
                elif method == 'ucb':
                    policy_probs = self.predict_ucb(X_val)
                elif method == 'bayesian':
                    action_values = {}
                    for action in self.actions:
                        if f"bayesian_{action}" in self.models:
                            model = self.models[f"bayesian_{action}"]
                            action_values[action] = model.predict_proba(X_val)[:, 1]
                        else:
                            action_values[action] = np.full(len(X_val), 0.1)
                    policy_probs = self.create_optimized_policy(action_values, 'softmax')
                
                score = self.evaluate_policy(policy_probs, actions_val, y_val, 'snips')
                print(f"{strategy_name}: SNIPS score = {score:.6f}")
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy_name
                    best_probs = policy_probs
                    
            except Exception as e:
                print(f"Ошибка в стратегии {strategy_name}: {e}")
        
        print(f"Лучшая RL стратегия: {best_strategy} с score {best_score:.6f}")
        self.best_strategy = best_strategy
        
        self._analyze_policy(best_probs, actions_val, y_val)
    
    def create_optimized_policy(self, action_values, method='softmax'):
        """Универсальный метод создания политики (для совместимости)"""
        values_df = pd.DataFrame(action_values)
        
        if method == 'softmax':
            temperature = 0.2
            exp_values = np.exp(values_df.values / temperature)
            probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
        else:
            exp_values = np.exp(values_df.values / 0.2)
            probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
        
        return probabilities
    
    def evaluate_policy(self, policy_probs, actual_actions, rewards, method='snips'):
        """Оценка политики"""
        action_map = {action: idx for idx, action in enumerate(self.actions)}
        logging_policy_prob = 1/3
        
        weights = []
        actual_rewards = []
        
        for i, action in enumerate(actual_actions):
            if action in action_map:
                action_idx = action_map[action]
                policy_prob = policy_probs[i, action_idx]
                weight = policy_prob / logging_policy_prob
                weights.append(weight)
                actual_rewards.append(rewards[i])
        
        weights = np.array(weights)
        actual_rewards = np.array(actual_rewards)
        
        if method == 'snips' and len(weights) > 0 and np.mean(weights) > 0:
            ips_estimate = np.mean(weights * actual_rewards)
            snips_estimate = ips_estimate / np.mean(weights)
            return snips_estimate
        
        return 0
    
    def predict(self, test_data):
        """Предсказание с лучшей RL стратегией"""
        print("Предсказание с RL методами...")
        X, _ = self.prepare_features(test_data, is_train=False)
        
        if 'thompson' in self.best_strategy:
            action_samples = self.predict_thompson_samples(X)
            if 'direct' in self.best_strategy:
                final_policy = self.create_thompson_policy(action_samples, 'direct')
            elif 'softmax' in self.best_strategy:
                final_policy = self.create_thompson_policy(action_samples, 'softmax')
            else:
                final_policy = self.create_thompson_policy(action_samples, 'ucb_thompson')
                
        elif 'ucb' in self.best_strategy:
            final_policy = self.predict_ucb(X)
            
        else:
            action_values = {}
            for action in self.actions:
                if f"bayesian_{action}" in self.models:
                    model = self.models[f"bayesian_{action}"]
                    action_values[action] = model.predict_proba(X)[:, 1]
                else:
                    action_values[action] = np.full(len(X), 0.1)
            final_policy = self.create_optimized_policy(action_values, 'softmax')
        
        result = pd.DataFrame({
            'id': test_data['id'].values,
            'p_mens_email': final_policy[:, 0],
            'p_womens_email': final_policy[:, 1],
            'p_no_email': final_policy[:, 2]
        })
        
        self._validate_submission(result)
        return result
    
    def _analyze_policy(self, policy_probs, actual_actions, rewards):
        """Анализ политики"""
        print("Анализ RL политики:")
        
        avg_probs = policy_probs.mean(axis=0)
        print(f"Средние вероятности: {dict(zip(self.actions, avg_probs))}")
        
        best_action_indices = np.argmax(policy_probs, axis=1)
        best_action_counts = pd.Series(best_action_indices).value_counts().sort_index()
        print("Распределение выбора действий:")
        for i, count in best_action_counts.items():
            print(f"  {self.actions[i]}: {count} ({count/len(policy_probs):.2%})")
    
    def _validate_submission(self, submission):
        """Валидация submission"""
        print("Проверка submission...")
        
        prob_sum = submission[['p_mens_email', 'p_womens_email', 'p_no_email']].sum(axis=1)
        assert np.allclose(prob_sum, 1.0, atol=1e-10), "Суммы вероятностей не равны 1"
        
        for col in ['p_mens_email', 'p_womens_email', 'p_no_email']:
            assert submission[col].between(0, 1).all(), f"Вероятности {col} вне диапазона"
        
        assert not submission.isnull().any().any(), "Есть NaN значения"
        
        print("✓ Submission прошел валидацию")
        print(f"Средние вероятности:")
        avg_probs = submission[['p_mens_email', 'p_womens_email', 'p_no_email']].mean()
        for action, prob in zip(self.actions, avg_probs):
            print(f"  {action}: {prob:.4f}")


def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """
    import os
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    predictions.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    # Загрузка данных
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    
    print("=== RL Пайплайн ===")
    print(f"Размер train: {train_data.shape}")
    
    print("\nРаспределение действий и rewards:")
    action_stats = train_data.groupby('segment').agg({
        'visit': ['count', 'mean'],
        'recency': 'mean',
        'history': 'mean'
    }).round(4)
    print(action_stats)
    
    # Обучение модели
    model = AdvancedThompsonSamplingBandit()
    model.fit(train_data)
    
    # Предсказание
    predictions = model.predict(test_data)
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(predictions)
    
    print("\nСтатистика финальной политики:")
    print(predictions[['p_mens_email', 'p_womens_email', 'p_no_email']].describe())
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    np.random.seed(42)
    main()
