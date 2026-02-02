import streamlit as st
import random
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple, Set, Union
from itertools import chain
import logging
import time
import os
import requests
from io import BytesIO

# ==============================================================================
# 1. إعدادات النظام
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("JordanLottery")

class LotteryConfig:
    MIN_NUM = 1
    MAX_NUM = 32
    DEFAULT_TICKET_SIZE = 6
    MIN_TICKET_SIZE = 6
    MAX_TICKET_SIZE = 10
    MAX_GENERATION_ATTEMPTS = 50000 
    STRICT_SHADOW_ATTEMPTS = 15000
    DEFAULT_SUM_TOLERANCE = 0.15
    MAX_BATCH_SIZE = 10
    
    # رابط ملف البيانات الافتراضي على GitHub
    # قم بتغيير هذا الرابط إلى رابط ملفك الخاص
    DEFAULT_GITHUB_URL = "https://raw.githubusercontent.com/MohamedOmariJo/jordan-lottery-app/main/history.xlsx"

def initialize_session_state():
    """تهيئة متغيرات الجلسة"""
    if 'history_df' not in st.session_state: 
        st.session_state.history_df = None
    if 'analyzer' not in st.session_state: 
        st.session_state.analyzer = None
    if 'generator' not in st.session_state: 
        st.session_state.generator = None
    if 'last_result' not in st.session_state: 
        st.session_state.last_result = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

# ==============================================================================
# 2. دالة جلب الملف من GitHub
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=3600)  # التخزين المؤقت لمدة ساعة
def load_from_github(github_url: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    تحميل ملف Excel من GitHub
    
    Parameters:
    -----------
    github_url : str
        رابط الملف على GitHub (يجب أن يكون رابط raw)
    
    Returns:
    --------
    Tuple[Optional[pd.DataFrame], str]
        DataFrame والرسالة
    """
    try:
        # تحويل رابط GitHub العادي إلى رابط raw
        if 'github.com' in github_url and '/blob/' in github_url:
            github_url = github_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
        
        # تحميل الملف
        response = requests.get(github_url, timeout=30)
        response.raise_for_status()
        
        # قراءة الملف من الذاكرة
        file_content = BytesIO(response.content)
        
        # تحديد نوع الملف وقراءته
        if github_url.endswith('.csv'):
            df = pd.read_csv(file_content)
        else:  # Excel
            df = pd.read_excel(file_content)
        
        # تنظيف أولي
        df.dropna(how='all', inplace=True)
        
        # التحقق من الأعمدة
        cols = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6']
        if not set(cols).issubset(df.columns):
            return None, "خطأ: الملف لا يحتوي على أعمدة الأرقام (N1...N6)"

        # تحويل الأرقام
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df.dropna(subset=cols, inplace=True)
        df['numbers'] = df[cols].values.tolist()
        
        # فلتر النطاق (1-32)
        def is_valid_draw(nums):
            return all(LotteryConfig.MIN_NUM <= int(n) <= LotteryConfig.MAX_NUM for n in nums)

        df = df[df['numbers'].apply(is_valid_draw)]
        
        if df.empty:
            return None, "خطأ: لا توجد بيانات صالحة (تأكد أن الأرقام بين 1 و 32)."

        # ترتيب الأرقام للتسهيل
        df['numbers'] = df['numbers'].apply(lambda x: sorted([int(n) for n in x]))
        
        # توحيد عمود المعرف
        if 'رقم السحب' in df.columns:
            df = df.rename(columns={'رقم السحب': 'draw_id'})
        elif 'DrawID' in df.columns:
            df = df.rename(columns={'DrawID': 'draw_id'})
        elif 'draw_id' not in df.columns:
            df['draw_id'] = range(1, len(df) + 1)
        
        # إعادة تعيين الفهرس
        df = df.reset_index(drop=True)
            
        return df, f"تم تحميل {len(df)} سحب بنجاح ✅"
        
    except requests.exceptions.RequestException as e:
        logger.error(f"GitHub loading error: {e}")
        return None, f"خطأ في الاتصال بـ GitHub: {str(e)}"
    except Exception as e:
        logger.error(f"Data processing error: {e}")
        return None, f"خطأ في معالجة الملف: {str(e)}"

# ==============================================================================
# 3. طبقة البيانات (تحميل محلي)
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_and_process_data(file_input: Union[str, object]) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        # تحديد نوع الملف (مسار أو ملف مرفوع)
        is_csv = False
        file_ref = file_input
        
        if isinstance(file_input, str):
            is_csv = file_input.endswith('.csv')
        else:
            is_csv = file_input.name.endswith('.csv')

        # القراءة
        if is_csv:
            df = pd.read_csv(file_ref)
        else:
            df = pd.read_excel(file_ref)
        
        # تنظيف أولي
        df.dropna(how='all', inplace=True)
        
        # التحقق من الأعمدة
        cols = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6']
        if not set(cols).issubset(df.columns):
            return None, "خطأ: الملف لا يحتوي على أعمدة الأرقام (N1...N6)"

        # تحويل الأرقام
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df.dropna(subset=cols, inplace=True)
        df['numbers'] = df[cols].values.tolist()
        
        # فلتر النطاق (1-32)
        def is_valid_draw(nums):
            return all(LotteryConfig.MIN_NUM <= int(n) <= LotteryConfig.MAX_NUM for n in nums)

        df = df[df['numbers'].apply(is_valid_draw)]
        
        if df.empty:
            return None, "خطأ: لا توجد بيانات صالحة (تأكد أن الأرقام بين 1 و 32)."

        # ترتيب الأرقام للتسهيل
        df['numbers'] = df['numbers'].apply(lambda x: sorted([int(n) for n in x]))
        
        # توحيد عمود المعرف
        if 'رقم السحب' in df.columns:
            df = df.rename(columns={'رقم السحب': 'draw_id'})
        elif 'DrawID' in df.columns:
            df = df.rename(columns={'DrawID': 'draw_id'})
        elif 'draw_id' not in df.columns:
            df['draw_id'] = range(1, len(df) + 1)
        
        # إعادة تعيين الفهرس
        df = df.reset_index(drop=True)
            
        return df, "Success"
        
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        return None, f"خطأ في معالجة الملف: {str(e)}"

# ==============================================================================
# 4. المحلل الإحصائي (Core Logic)
# ==============================================================================
class LotteryAnalyzer:
    def __init__(self, history_df: pd.DataFrame):
        self.history_df = history_df
        self.past_draws_sets = [set(nums) for nums in history_df['numbers']]
        self.draw_map = {int(row['draw_id']): row['numbers'] for _, row in history_df.iterrows()}
        
        self.number_to_draws_index = defaultdict(set)
        for idx, draw_set in enumerate(self.past_draws_sets):
            for num in draw_set:
                self.number_to_draws_index[num].add(idx)
        
        all_numbers = list(chain.from_iterable(history_df['numbers']))
        self.frequency = Counter(all_numbers)
        self.total_draws = len(history_df)
        
        all_sums = [sum(nums) for nums in history_df['numbers']]
        self.global_avg_sum = sum(all_sums) / len(all_sums) if all_sums else 0
        
        sorted_nums = sorted(range(LotteryConfig.MIN_NUM, LotteryConfig.MAX_NUM + 1), 
                           key=lambda x: self.frequency.get(x, 0), reverse=True)
        self.hot_pool = set(sorted_nums[:16])
        self.cold_pool = set(sorted_nums[16:])

    def calculate_custom_average(self, mode: str, param1: int = 0, param2: int = 0) -> tuple:
        df = self.history_df.copy()
        if mode == "Last N Draws":
            if param1 > len(df): 
                param1 = len(df)
            df = df.iloc[-param1:]
        elif mode == "Specific Range":
            df = df[(df['draw_id'] >= param1) & (df['draw_id'] <= param2)]
        
        if df.empty: 
            return self.global_avg_sum, []
        sums = [sum(nums) for nums in df['numbers']]
        avg = sum(sums) / len(sums) if sums else 0
        return avg, sums

    def get_ticket_profile(self, ticket: List[int]) -> str:
        hot_count = sum(1 for n in ticket if n in self.hot_pool)
        total = len(ticket)
        if hot_count >= total * 0.7: 
            return "🔥 ساخنة"
        elif hot_count <= total * 0.3: 
            return "❄️ باردة"
        else: 
            return "⚖️ متوازنة"
    
    def get_numbers_from_draw(self, draw_id: int) -> Optional[List[int]]:
        return self.draw_map.get(int(draw_id))

    def check_matches_history(self, ticket_numbers: List[int]) -> Dict[int, List[Dict]]:
        matches_found = {6: [], 5: [], 4: []}
        ticket_set = set(ticket_numbers)
        for draw_id, draw_nums in self.draw_map.items():
            intersection = ticket_set & set(draw_nums)
            count = len(intersection)
            if count in matches_found:
                matches_found[count].append({
                    'draw_id': draw_id, 
                    'matched_nums': sorted(list(intersection))
                })
        return matches_found

    def get_numbers_frequency_stats(self, ticket_numbers: List[int]) -> pd.DataFrame:
        stats = []
        for num in ticket_numbers:
            count = self.frequency.get(num, 0)
            stats.append({'الرقم': num, 'عدد مرات الظهور': count})
        return pd.DataFrame(stats).sort_values(by='عدد مرات الظهور', ascending=False)

    def analyze_sequences_history(self, ticket_numbers: List[int]) -> Dict:
        sorted_nums = sorted(ticket_numbers)
        sequences = []
        if not sorted_nums: 
            return {}
        
        temp_seq = [sorted_nums[0]]
        for i in range(1, len(sorted_nums)):
            if sorted_nums[i] == sorted_nums[i-1] + 1:
                temp_seq.append(sorted_nums[i])
            else:
                if len(temp_seq) >= 2: 
                    sequences.append(temp_seq)
                temp_seq = [sorted_nums[i]]
        
        if len(temp_seq) >= 2: 
            sequences.append(temp_seq)
        
        if not sequences: 
            return {}
        
        results = {}
        for seq in sequences:
            seq_tuple = tuple(seq)
            seq_set = set(seq)
            full_count = sum(1 for draw_set in self.past_draws_sets if seq_set.issubset(draw_set))
            full_draws = [
                self.history_df.iloc[idx]['draw_id'] 
                for idx in range(len(self.past_draws_sets)) 
                if seq_set.issubset(self.past_draws_sets[idx])
            ]
            
            sub_dict = {}
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i+1])
                pair_set = set(pair)
                pair_count = sum(1 for draw_set in self.past_draws_sets if pair_set.issubset(draw_set))
                pair_draws = [
                    self.history_df.iloc[idx]['draw_id'] 
                    for idx in range(len(self.past_draws_sets)) 
                    if pair_set.issubset(self.past_draws_sets[idx])
                ]
                sub_dict[pair] = {'count': pair_count, 'draws': pair_draws}
            
            results[seq_tuple] = {'full_count': full_count, 'full_draws': full_draws, 'sub': sub_dict}
        
        return results

# ==============================================================================
# 5. مولد التذاكر
# ==============================================================================
class TicketGenerator:
    def __init__(self, analyzer: LotteryAnalyzer):
        self.analyzer = analyzer
    
    def _count_sequences(self, nums: List[int]) -> int:
        if len(nums) < 2: 
            return 0
        sorted_nums = sorted(nums)
        sequences_count = 0
        i = 0
        while i < len(sorted_nums) - 1:
            if sorted_nums[i+1] == sorted_nums[i] + 1:
                sequences_count += 1
                i += 2
            else:
                i += 1
        return sequences_count

    def _count_shadows(self, nums: List[int]) -> int:
        nums_set = set(nums)
        shadows_count = 0
        for num in nums:
            if (num - 1 in nums_set) or (num + 1 in nums_set):
                shadows_count += 1
        return shadows_count

    def _count_odd(self, nums: List[int]) -> int:
        return sum(1 for n in nums if n % 2 == 1)

    def _check_sum_condition(self, nums: List[int], target_avg: float) -> bool:
        s = sum(nums)
        tolerance = target_avg * LotteryConfig.DEFAULT_SUM_TOLERANCE
        return abs(s - target_avg) <= tolerance

    def _count_match(self, ticket_set: set, draw_set: set) -> int:
        return len(ticket_set & draw_set)

    def _validate_criteria(self, criteria: Dict) -> List[str]:
        errors = []
        size = criteria['size']
        if criteria['sequences_count'] >= size:
            errors.append("عدد المتتاليات يجب أن يكون أقل من حجم التذكرة")
        if criteria['odd_count'] > size:
            errors.append("عدد الأرقام الفردية يجب ألا يتجاوز حجم التذكرة")
        if criteria['shadows_count'] > size:
            errors.append("عدد الظلال يجب ألا يتجاوز حجم التذكرة")
        if criteria.get('include_count', 0) > size:
            errors.append("عدد الأرقام المطلوبة من السحب يجب ألا يتجاوز حجم التذكرة")
        return errors

    def generate_single(self, criteria: Dict, attempt_limit: int = None) -> Optional[List[int]]:
        if attempt_limit is None:
            attempt_limit = LotteryConfig.MAX_GENERATION_ATTEMPTS
        
        size = criteria['size']
        req_seq = criteria['sequences_count']
        req_odd = criteria['odd_count']
        req_sha = criteria['shadows_count']
        anti = criteria['anti_match_limit']
        strategy = criteria.get('strategy', 'balanced')
        sum_check = criteria.get('sum_near_avg', False)
        target_avg = criteria.get('target_average', self.analyzer.global_avg_sum)
        
        include_draw = criteria.get('include_from_draw')
        include_count = criteria.get('include_count', 0)
        forced_numbers = []
        
        if include_draw and include_count > 0:
            past_nums = self.analyzer.get_numbers_from_draw(include_draw)
            if past_nums:
                forced_numbers = random.sample(past_nums, min(include_count, len(past_nums)))
        
        base_pool = set(range(LotteryConfig.MIN_NUM, LotteryConfig.MAX_NUM + 1))
        available_pool = base_pool - set(forced_numbers)
        
        if strategy == 'hot':
            candidates = sorted(available_pool, key=lambda x: self.analyzer.frequency.get(x, 0), reverse=True)
            candidates = candidates[:24]
        elif strategy == 'cold':
            candidates = sorted(available_pool, key=lambda x: self.analyzer.frequency.get(x, 0))
            candidates = candidates[:24]
        else:
            candidates = list(available_pool)
        
        strict_shadow_mode = (req_sha >= 4)
        shadow_attempts = LotteryConfig.STRICT_SHADOW_ATTEMPTS if strict_shadow_mode else attempt_limit
        
        for attempt in range(shadow_attempts):
            needed = size - len(forced_numbers)
            if needed <= 0:
                ticket = forced_numbers[:]
            else:
                picked = random.sample(candidates, needed)
                ticket = forced_numbers + picked
            
            ticket_set = set(ticket)
            
            if self._count_sequences(ticket) != req_seq:
                continue
            if self._count_odd(ticket) != req_odd:
                continue
            if self._count_shadows(ticket) != req_sha:
                continue
            
            if sum_check and not self._check_sum_condition(ticket, target_avg):
                continue
            
            violates = any(
                self._count_match(ticket_set, draw_set) >= anti 
                for draw_set in self.analyzer.past_draws_sets
            )
            
            if violates:
                continue
            
            return sorted(ticket)
        
        if strict_shadow_mode and shadow_attempts < attempt_limit:
            for attempt in range(attempt_limit - shadow_attempts):
                needed = size - len(forced_numbers)
                if needed <= 0:
                    ticket = forced_numbers[:]
                else:
                    picked = random.sample(candidates, needed)
                    ticket = forced_numbers + picked
                
                ticket_set = set(ticket)
                
                if self._count_sequences(ticket) != req_seq:
                    continue
                if self._count_odd(ticket) != req_odd:
                    continue
                
                if sum_check and not self._check_sum_condition(ticket, target_avg):
                    continue
                
                violates = any(
                    self._count_match(ticket_set, draw_set) >= anti 
                    for draw_set in self.analyzer.past_draws_sets
                )
                
                if violates:
                    continue
                
                return sorted(ticket)
        
        return None

    def generate_batch(self, criteria: Dict, count: int) -> Dict:
        validation_errors = self._validate_criteria(criteria)
        if validation_errors:
            return {'status': 'validation_error', 'errors': validation_errors, 'tickets': [], 'generated': 0}
        
        if count > LotteryConfig.MAX_BATCH_SIZE:
            return {
                'status': 'validation_error', 
                'errors': [f"الحد الأقصى {LotteryConfig.MAX_BATCH_SIZE} تذاكر"], 
                'tickets': [], 
                'generated': 0
            }
        
        tickets = []
        seen = set()
        
        for i in range(count):
            ticket = self.generate_single(criteria)
            if ticket is None:
                break
            
            ticket_tuple = tuple(ticket)
            if ticket_tuple in seen:
                continue
            
            seen.add(ticket_tuple)
            
            analysis = {
                'sum': sum(ticket),
                'sequences': self._count_sequences(ticket),
                'shadows': self._count_shadows(ticket),
                'odd': self._count_odd(ticket),
                'profile': self.analyzer.get_ticket_profile(ticket)
            }
            
            tickets.append({'id': i+1, 'numbers': ticket, 'analysis': analysis})
        
        generated = len(tickets)
        
        if generated == 0:
            return {
                'status': 'failed', 
                'errors': ['فشل توليد أي تذكرة. جرّب تخفيف الشروط.'], 
                'tickets': [], 
                'generated': 0
            }
        
        if generated < count:
            return {
                'status': 'partial_success', 
                'tickets': tickets, 
                'generated': generated, 
                'errors': [f"تم توليد {generated} من أصل {count} تذاكر. جرّب تخفيف الشروط."]
            }
        
        return {'status': 'success', 'tickets': tickets, 'generated': generated, 'errors': []}

    def estimate_success_probability(self, criteria: Dict) -> Dict:
        validation_errors = self._validate_criteria(criteria)
        if validation_errors:
            return {'probability': 0, 'advice': "الشروط غير صحيحة"}
        
        sample_attempts = 1000
        success_count = 0
        
        for _ in range(sample_attempts):
            ticket = self.generate_single(criteria, attempt_limit=100)
            if ticket is not None:
                success_count += 1
        
        probability = (success_count / sample_attempts) * 100
        
        if probability >= 10:
            advice = "ممتاز - الشروط واقعية جداً"
        elif probability >= 5:
            advice = "جيد - فرصة معقولة"
        elif probability >= 1:
            advice = "صعب - قد يستغرق وقتاً"
        else:
            advice = "صعب جداً - فكر في تخفيف الشروط"
        
        return {'probability': round(probability, 2), 'advice': advice}

# ==============================================================================
# 6. واجهة المستخدم
# ==============================================================================
def main():
    st.set_page_config(page_title="توقعات اليانصيب الأردني", page_icon="🎰", layout="wide")
    
    st.markdown("""
        <style>
            .main > div { padding-top: 2rem; }
            .stButton>button { width: 100%; }
            .footer { text-align: center; margin-top: 50px; color: gray; font-size: 0.9em; }
            
            .logo-container {
                text-align: center;
                margin-bottom: 20px;
            }
            
            .logo-container img {
                width: 120px;
                height: 120px;
                border-radius: 50%;
                box-shadow: 0 4px 20px rgba(255, 0, 0, 0.3);
                border: 4px solid #ffffff;
                background: white;
                padding: 5px;
            }
            
            .fancy-title {
                text-align: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 25px 20px;
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
                margin-bottom: 30px;
                border: 3px solid #ffffff;
            }
            
            .fancy-title h1 {
                color: #ffffff;
                font-size: 2.2em;
                font-weight: 800;
                margin: 0;
                padding: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                letter-spacing: 1px;
                font-family: 'Cairo', 'Segoe UI', Tahoma, sans-serif;
                line-height: 1.3;
                word-wrap: break-word;
            }
            
            .fancy-title .emoji {
                font-size: 1em;
                margin: 0 10px;
                display: inline;
                vertical-align: middle;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.1); }
            }
            
            @media (max-width: 768px) {
                .logo-container img {
                    width: 90px;
                    height: 90px;
                }
                .fancy-title {
                    padding: 20px 10px;
                }
                .fancy-title h1 {
                    font-size: 1.4em;
                }
                .fancy-title .emoji {
                    font-size: 0.9em;
                    margin: 0 6px;
                }
            }
            
            @media (max-width: 480px) {
                .logo-container img {
                    width: 70px;
                    height: 70px;
                }
                .fancy-title h1 {
                    font-size: 1.1em;
                }
                .fancy-title .emoji {
                    margin: 0 4px;
                }
            }
        </style>
        
        <div class="logo-container">
            <img src="https://raw.githubusercontent.com/MohamedOmariJo/jordan-lottery-app/main/lotto_logo.png" alt="Jordan Lotto Logo">
        </div>
        
        <div class="fancy-title">
            <h1><span class="emoji">🎯</span>القناص لفحص وتوليد تذاكر لوتري الأردن<span class="emoji">🎰</span></h1>
        </div>
    """, unsafe_allow_html=True)
    
    # st.title("🎰 نظام محمد العمري لفحص وتوليد تذاكر لوتري الأردن")
    initialize_session_state()
    
    # تحميل البيانات تلقائياً عند أول تشغيل
    if not st.session_state.data_loaded:
        with st.spinner("جاري تحميل البيانات من GitHub..."):
            df, msg = load_from_github(LotteryConfig.DEFAULT_GITHUB_URL)
            
            if df is not None:
                st.session_state.history_df = df
                st.session_state.analyzer = LotteryAnalyzer(df)
                st.session_state.generator = TicketGenerator(st.session_state.analyzer)
                st.session_state.data_loaded = True
                st.success(msg)
            else:
                st.error("⚠️ " + msg)
                st.warning("يمكنك تحميل ملف البيانات يدوياً من الشريط الجانبي")
    
    # قسم تحميل البيانات في الشريط الجانبي (اختياري)
    with st.sidebar.expander("🔄 تحديث البيانات", expanded=False):
        st.info("البيانات محملة تلقائياً من GitHub. يمكنك إعادة تحميلها أو رفع ملف جديد.")
        
        if st.button("🔄 إعادة تحميل من GitHub"):
            # مسح الذاكرة المؤقتة
            load_from_github.clear()
            st.session_state.data_loaded = False
            st.rerun()
        
        st.markdown("---")
        st.markdown("**أو ارفع ملف محلي:**")
        
        uploaded_file = st.file_uploader(
            "اختر ملف Excel/CSV:", 
            type=['xlsx', 'xls', 'csv'],
            help="ارفع ملف يحتوي على بيانات السحوبات السابقة",
            key="file_uploader"
        )
        
        if uploaded_file:
            with st.spinner("جاري تحميل البيانات..."):
                df, msg = load_and_process_data(uploaded_file)
                
                if df is not None:
                    st.session_state.history_df = df
                    st.session_state.analyzer = LotteryAnalyzer(df)
                    st.session_state.generator = TicketGenerator(st.session_state.analyzer)
                    st.session_state.data_loaded = True
                    st.success(f"تم تحميل {len(df)} سحب بنجاح ✅")
                    st.rerun()
                else:
                    st.error(msg)
    
    # التحقق من تحميل البيانات
    if st.session_state.history_df is None:
        st.warning("⚠️ لم يتم تحميل البيانات بنجاح")
        st.info("""
        **حاول:**
        - التحقق من اتصالك بالإنترنت
        - التأكد من أن رابط GitHub صحيح في ملف الكود
        - رفع ملف البيانات يدوياً من الشريط الجانبي
        
        **متطلبات الملف:**
        - يجب أن يحتوي على أعمدة: N1, N2, N3, N4, N5, N6
        - الأرقام يجب أن تكون بين 1 و 32
        - يمكن استخدام ملفات Excel (.xlsx, .xls) أو CSV
        """)
        st.stop()
    
    analyzer = st.session_state.analyzer
    generator = st.session_state.generator
    
    # عرض معلومات البيانات
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 معلومات البيانات")
    st.sidebar.metric("📈 إجمالي السحوبات", analyzer.total_draws)
    st.sidebar.metric("📊 متوسط المجموع", f"{analyzer.global_avg_sum:.1f}")
    
    # معلومات التحديث
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **ملاحظة:** يتم تحديث البيانات تلقائياً كل أحد وأربعاء")
    
    # --------------------------------------------------------
    # Tabs
    # --------------------------------------------------------
    tab1, tab2 = st.tabs(["🎲 مولد التذاكر", "🕵️ فحص تذكرة"])
    
    # --------------------------------------------------------
    # Tab 1: Generator
    # --------------------------------------------------------
    with tab1:
        st.markdown("""
            <style>
                .header-with-logo {
                    display: inline-flex;
                    align-items: center;
                    gap: 10px;
                }
                .header-with-logo img {
                    width: 35px;
                    height: 35px;
                    border-radius: 50%;
                    vertical-align: middle;
                }
            </style>
        """, unsafe_allow_html=True)
        st.markdown(
            '<h3 class="header-with-logo">'
            '<img src="https://raw.githubusercontent.com/MohamedOmariJo/jordan-lottery-app/main/lotto_logo.png" alt="Logo">'
            ' توليد تذاكر ذكية</h3>',
            unsafe_allow_html=True
        )
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with st.container(border=True):
                st.markdown("**📐 الاستراتيجية والمتوسط**")
                strategy = st.selectbox("اختر الاستراتيجية:", ["⚖️ كرات مختلطة القوة", "🔥 كرات ساخنة (الأكثر ظهوراً)", "❄️ كرات باردة (الأقل ظهوراً)"])
                strategy_map = {"⚖️ كرات مختلطة القوة": "balanced", "🔥 كرات ساخنة (الأكثر ظهوراً)": "hot", "❄️ كرات باردة (الأقل ظهوراً)": "cold"}
                
                avg_mode = st.radio("حساب المتوسط:", ["Global", "Last N Draws", "Specific Range"], horizontal=True)
                
                avg_chk = False
                target_avg_val = analyzer.global_avg_sum
                
                if avg_mode == "Global":
                    st.info(f"المتوسط العام: {analyzer.global_avg_sum:.1f}")
                    avg_chk = st.checkbox("تطبيق شرط المتوسط")
                    if avg_chk: 
                        target_avg_val = analyzer.global_avg_sum
                
                elif avg_mode == "Last N Draws":
                    n_draws = st.number_input("عدد السحوبات الأخيرة:", 1, analyzer.total_draws, 50)
                    avg_val, sums = analyzer.calculate_custom_average("Last N Draws", n_draws)
                    st.info(f"متوسط آخر {n_draws} سحب: {avg_val:.1f}")
                    avg_chk = st.checkbox("تطبيق شرط المتوسط")
                    if avg_chk: 
                        target_avg_val = avg_val
                
                else:
                    c1, c2 = st.columns(2)
                    from_draw = c1.number_input("من سحب:", 1, analyzer.total_draws, 1)
                    to_draw = c2.number_input("إلى سحب:", 1, analyzer.total_draws, analyzer.total_draws)
                    avg_val, sums = analyzer.calculate_custom_average("Specific Range", from_draw, to_draw)
                    st.info(f"المتوسط في النطاق: {avg_val:.1f}")
                    avg_chk = st.checkbox("تطبيق شرط المتوسط")
                    if avg_chk: 
                        target_avg_val = avg_val

            with st.container(border=True):
                st.markdown("**🎯 معايير التذكرة**")
                t_count = st.number_input("عدد التذاكر المراد توليدها", 1, 10, 3)
                t_size = st.slider("حجم التذكرة", 6, 10, 6)
                odd = st.number_input("عدد الكرات التي تحمل أرقام فردية في كل تذكرة مراد توليدها", 0, t_size, t_size//2)
                seq = st.number_input("عدد المتتاليات في كل تذكرة مراد توليدها", 0, t_size-1, 0)
                sha = st.number_input("عدد الظلال في كل تذكرة مراد توليدها", 0, 3, 1)

            with st.container(border=True):
                st.markdown("**🔄 تكرار صارم (Pivot)**")
                use_past = st.checkbox("تثبيت أرقام من سحب سابق")
                inc_draw = None
                inc_cnt = 0
                
                if use_past:
                    c1, c2 = st.columns(2)
                    inc_draw = c1.number_input("رقم السحب", 1, analyzer.total_draws, analyzer.total_draws)
                    inc_cnt = c2.number_input("عدد الأرقام", 1, min(6, t_size), 1)
                    past_nums = analyzer.get_numbers_from_draw(inc_draw)
                    if past_nums: 
                        st.caption(f"أرقام السحب {inc_draw}: {past_nums}")

            st.markdown("---")
            anti = st.slider("تجنب تطابق (عدد أرقام) مع أي سحب سابق", 3, t_size, 5)

            criteria = {
                'size': t_size, 
                'sequences_count': seq, 
                'odd_count': odd, 
                'shadows_count': sha, 
                'anti_match_limit': anti, 
                'sum_near_avg': avg_chk,
                'target_average': target_avg_val,
                'include_from_draw': inc_draw if use_past else None, 
                'include_count': inc_cnt if use_past else 0,
                'strategy': strategy_map[strategy]
            }

            if st.button("🔍 فحص الجدوى"):
                with st.spinner("جاري المحاكاة..."):
                    est = generator.estimate_success_probability(criteria)
                    color = "green" if est['probability'] > 5 else "red"
                    st.markdown(f"**النسبة:** :{color}[{est['probability']}%] ({est['advice']})")

            if st.button("🚀 توليد الآن", type="primary", use_container_width=True):
                with st.spinner("جاري بدء المحرك..."):
                    st.session_state.last_result = generator.generate_batch(criteria, t_count)

        with col2:
            if st.session_state.last_result:
                res = st.session_state.last_result
                
                if res['status'] == 'validation_error':
                    st.error("خطأ:")
                    for e in res['errors']:
                        st.write(f"- {e}")
                
                elif res['status'] == 'failed':
                    st.error("فشل التوليد.")
                    st.write("الأسباب:", res['errors'])
                
                else:
                    if res['status'] == 'partial_success': 
                        st.warning(f"تم توليد {res['generated']} تذاكر فقط.")
                    else: 
                        st.success(f"تم توليد {res['generated']} تذاكر بنجاح!")
                    
                    for t in res['tickets']:
                        with st.expander(f"🎫 تذكرة #{t['id']} - {t['analysis']['profile']}", expanded=True):
                            st.markdown(
                                "".join([
                                    f"<span style='display:inline-block; background:#dcfce7; color:#166534; "
                                    f"padding:5px 10px; margin:2px; border-radius:50%; font-weight:bold; "
                                    f"border:1px solid #166534'>{n}</span>" 
                                    for n in t['numbers']
                                ]), 
                                unsafe_allow_html=True
                            )
                            
                            ca, cb, cc = st.columns(3)
                            ca.caption(f"المجموع: {t['analysis']['sum']}")
                            cb.caption(f"المتتاليات: {t['analysis']['sequences']}")
                            cc.caption(f"الظلال: {t['analysis']['shadows']}")
                            
                            if use_past and inc_draw:
                                draw_nums = set(analyzer.get_numbers_from_draw(inc_draw))
                                matches = set(t['numbers']) & draw_nums
                                color = "green" if len(matches) == inc_cnt else "red"
                                st.markdown(
                                    f":{color}[✅ المطلوب: {inc_cnt} | "
                                    f"🎯 المحقق: {len(matches)} ({list(matches)})]"
                                )

    # --------------------------------------------------------
    # Tab 2: Checker
    # --------------------------------------------------------
    with tab2:
        st.subheader("🕵️ فحص تذكرة تاريخياً")
        c_check1, c_check2 = st.columns([1, 2])
        
        with c_check1:
            chk_size = st.radio(
                "حدد حجم التذكرة للفحص:", 
                [6, 7, 8, 9, 10], 
                horizontal=True
            )
        
        with c_check2:
            chk_numbers = st.multiselect(
                f"اختر {chk_size} أرقام بدقة:",
                options=list(range(1, 33)),
                max_selections=chk_size,
                help="اختر الأرقام التي تريد فحصها تاريخياً"
            )
        
        if st.button("🔎 ابدأ الفحص الشامل", type="primary", use_container_width=True):
            if len(chk_numbers) != chk_size:
                st.error(f"⚠️ يجب اختيار {chk_size} أرقام بالضبط. أنت اخترت {len(chk_numbers)}.")
            else:
                sorted_chk = sorted(chk_numbers)
                st.success(f"جاري فحص التذكرة: {sorted_chk}")
                
                # 1. Matches
                matches = analyzer.check_matches_history(sorted_chk)
                st.markdown("### 1️⃣ سجل التطابقات (Matches)")
                found_any = False
                
                for count in [6, 5, 4]:
                    res_list = matches[count]
                    if res_list:
                        found_any = True
                        with st.expander(
                            f"🌟 تطابق {count} أرقام (عدد المرات: {len(res_list)})", 
                            expanded=True
                        ):
                            for item in res_list:
                                st.markdown(
                                    f"- **سحب رقم {item['draw_id']}:** "
                                    f"الأرقام المتطابقة {item['matched_nums']}"
                                )
                
                if not found_any: 
                    st.info("✅ هذه التذكرة نظيفة! (لم تحقق 4,5,6 سابقاً)")

                st.divider()

                # 2. Frequency
                st.markdown("### 2️⃣ تحليل تكرار الأرقام")
                freq_df = analyzer.get_numbers_frequency_stats(sorted_chk)
                col_f1, col_f2 = st.columns([1, 2])
                
                with col_f1: 
                    st.dataframe(freq_df, hide_index=True, use_container_width=True)
                
                with col_f2: 
                    st.bar_chart(
                        freq_df.set_index('الرقم')['عدد مرات الظهور'], 
                        color="#166534"
                    )

                st.divider()

                # 3. Sequences
                st.markdown("### 3️⃣ فحص المتتاليات")
                seq_results = analyzer.analyze_sequences_history(sorted_chk)
                
                if not seq_results: 
                    st.write("🔹 لا توجد متتاليات.")
                else:
                    for seq_tuple, data in seq_results.items():
                        st.markdown(f"#### 🔗 المتتالية: `{seq_tuple}`")
                        st.write(f"- **ظهرت كاملة:** {data['full_count']} مرة.")
                        
                        if data['full_draws']:
                            st.caption(f"📍 في السحوبات: {data['full_draws']}")
                        
                        if data['sub']:
                            st.write("- **الأجزاء الثنائية:**")
                            for sub_pair, sub_data in data['sub'].items():
                                st.write(f"  - الثنائية `{sub_pair}` ظهرت: **{sub_data['count']}** مرة.")
                                if sub_data['draws']:
                                    with st.expander(f"عرض سحوبات {sub_pair}"):
                                        st.write(f"{sub_data['draws']}")
                        
                        st.markdown("---")

    st.markdown(
        """<div class="footer">برمجة وتطوير: <b>محمد العمري</b></div>""", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
