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
    # رابط ملف البيانات على GitHub - عدّل هذا الرابط حسب مستودعك
    GITHUB_DATA_URL = "https://raw.githubusercontent.com/MohamedOmariJo/jordan-lottery-app/main/history.xlsx"

def initialize_session_state():
    """تهيئة متغيرات الجلسة"""
    if 'history_df' not in st.session_state: st.session_state.history_df = None
    if 'analyzer' not in st.session_state: st.session_state.analyzer = None
    if 'generator' not in st.session_state: st.session_state.generator = None
    if 'last_result' not in st.session_state: st.session_state.last_result = None
    if 'auto_loaded' not in st.session_state: st.session_state.auto_loaded = False

# ==============================================================================
# 2. طبقة البيانات (تحميل حقيقي فقط)
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_from_github(url: str = None) -> Tuple[Optional[pd.DataFrame], str]:
    """تحميل البيانات من GitHub"""
    try:
        if url is None:
            url = LotteryConfig.GITHUB_DATA_URL
        
        # تحميل الملف من GitHub
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # قراءة الملف من الذاكرة
        file_content = BytesIO(response.content)
        df = pd.read_excel(file_content)
        
        # تنظيف أولي
        df.dropna(how='all', inplace=True)
        
        # التحقق من الأعمدة
        cols = ['N1','N2','N3','N4','N5','N6']
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
            
        return df, f"✅ تم تحميل {len(df)} سحب من GitHub"
        
    except requests.exceptions.RequestException as e:
        logger.error(f"GitHub loading error: {e}")
        return None, f"خطأ في الاتصال بـ GitHub: {str(e)}"
    except Exception as e:
        logger.error(f"Data processing error: {e}")
        return None, f"خطأ في معالجة الملف: {str(e)}"

@st.cache_data(show_spinner=False)
def load_and_process_data(file_input: Union[str, st.runtime.uploaded_file_manager.UploadedFile]) -> Tuple[Optional[pd.DataFrame], str]:
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
        cols = ['N1','N2','N3','N4','N5','N6']
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

        original_len = len(df)
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
            
        return df, "Success"
        
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        return None, f"خطأ في معالجة الملف: {str(e)}"

# تمت إزالة MockDataGenerator نهائياً لضمان المصداقية

# ==============================================================================
# 3. المحلل الإحصائي (Core Logic)
# ==============================================================================
class LotteryAnalyzer:
    def __init__(self, history_df: pd.DataFrame):
        self.history_df = history_df
        self.past_draws_sets = [set(nums) for nums in history_df['numbers']]
        self.draw_map = {row['draw_id']: row['numbers'] for _, row in history_df.iterrows()}
        
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
                           key=lambda x: self.frequency[x], reverse=True)
        self.hot_pool = set(sorted_nums[:16])
        self.cold_pool = set(sorted_nums[16:])

    def calculate_custom_average(self, mode: str, param1: int = 0, param2: int = 0) -> tuple:
        df = self.history_df.copy()
        if mode == "Last N Draws":
            if param1 > len(df): param1 = len(df)
            df = df.iloc[-param1:]
        elif mode == "Specific Range":
            df = df[(df['draw_id'] >= param1) & (df['draw_id'] <= param2)]
        
        if df.empty: return self.global_avg_sum, []
        sums = [sum(nums) for nums in df['numbers']]
        avg = sum(sums) / len(sums) if sums else 0
        return avg, sums

    def get_ticket_profile(self, ticket: List[int]) -> str:
        hot_count = sum(1 for n in ticket if n in self.hot_pool)
        total = len(ticket)
        if hot_count >= total * 0.7: return "🔥 ساخنة"
        elif hot_count <= total * 0.3: return "❄️ باردة"
        else: return "⚖️ متوازنة"
    
    def get_numbers_from_draw(self, draw_id: int) -> Optional[List[int]]:
        return self.draw_map.get(draw_id)

    # --- دوال الفحص التاريخي ---
    def check_matches_history(self, ticket_numbers: List[int]) -> Dict[int, List[Dict]]:
        matches_found = {6: [], 5: [], 4: []}
        ticket_set = set(ticket_numbers)
        for draw_id, draw_nums in self.draw_map.items():
            intersection = ticket_set & set(draw_nums)
            count = len(intersection)
            if count in matches_found:
                matches_found[count].append({'draw_id': draw_id, 'matched_nums': sorted(list(intersection))})
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
        if not sorted_nums: return {}
        
        temp_seq = [sorted_nums[0]]
        for i in range(1, len(sorted_nums)):
            if sorted_nums[i] == sorted_nums[i-1] + 1:
                temp_seq.append(sorted_nums[i])
            else:
                if len(temp_seq) >= 2: sequences.append(temp_seq)
                temp_seq = [sorted_nums[i]]
        if len(temp_seq) >= 2: sequences.append(temp_seq)
        
        results = {}
        for seq in sequences:
            seq_tuple = tuple(seq)
            results[seq_tuple] = {
                'full_count': 0, 
                'full_draws': [], 
                'sub': {}
            }
            seq_set = set(seq)
            
            for draw_id, draw_nums in self.draw_map.items():
                draw_set = set(draw_nums)
                if seq_set.issubset(draw_set):
                    results[seq_tuple]['full_count'] += 1
                    results[seq_tuple]['full_draws'].append(draw_id)
            
            if len(seq) > 2:
                for i in range(len(seq) - 1):
                    sub_pair = (seq[i], seq[i+1])
                    sub_set = set(sub_pair)
                    results[seq_tuple]['sub'][sub_pair] = {'count': 0, 'draws': []}
                    
                    for draw_id, draw_nums in self.draw_map.items():
                        draw_set = set(draw_nums)
                        if sub_set.issubset(draw_set):
                            results[seq_tuple]['sub'][sub_pair]['count'] += 1
                            results[seq_tuple]['sub'][sub_pair]['draws'].append(draw_id)
                            
        return results

# ==============================================================================
# 4. المدقق والمولد
# ==============================================================================
class TicketValidator:
    @staticmethod
    def count_sequences(numbers: List[int]) -> int:
        sorted_nums = sorted(numbers)
        return sum(1 for i in range(len(sorted_nums) - 1) if sorted_nums[i + 1] == sorted_nums[i] + 1)

    @staticmethod
    def count_shadow_occurrences(numbers: List[int]) -> int:
        units = [n % 10 for n in numbers]
        return sum(count - 1 for count in Counter(units).values() if count >= 2)

    @staticmethod
    def check_anti_match_optimized(ticket_set: set, analyzer: 'LotteryAnalyzer', limit: int) -> bool:
        candidate_indices = set()
        for num in ticket_set:
            candidate_indices.update(analyzer.number_to_draws_index.get(num, set()))
        for idx in candidate_indices:
            if len(ticket_set & analyzer.past_draws_sets[idx]) >= limit:
                return False
        return True

    @staticmethod
    def analyze_ticket(numbers: List[int], analyzer: Optional['LotteryAnalyzer'] = None) -> Dict:
        analysis = {
            'sum': sum(numbers),
            'sequences': TicketValidator.count_sequences(numbers),
            'shadows': TicketValidator.count_shadow_occurrences(numbers),
            'odd': sum(1 for n in numbers if n % 2 != 0),
        }
        if analyzer:
            analysis['profile'] = analyzer.get_ticket_profile(numbers)
        return analysis

class TicketGenerator:
    def __init__(self, analyzer: LotteryAnalyzer):
        self.analyzer = analyzer
        self.full_pool = list(range(LotteryConfig.MIN_NUM, LotteryConfig.MAX_NUM + 1))

    def _validate_criteria(self, criteria: Dict) -> List[str]:
        errors = []
        size = criteria.get('size', 6)
        inc_draw = criteria.get('include_from_draw')
        inc_cnt = criteria.get('include_count', 0)
        if inc_draw is not None and inc_cnt > 0:
            if inc_cnt > size: errors.append(f"❌ لا يمكن اقتباس {inc_cnt} أرقام في تذكرة حجمها {size}")
            if not self.analyzer.get_numbers_from_draw(inc_draw): errors.append(f"❌ السحب {inc_draw} غير موجود")
        if not (LotteryConfig.MIN_TICKET_SIZE <= size <= LotteryConfig.MAX_TICKET_SIZE):
            errors.append(f"❌ الحجم {size} غير صالح")
        return errors

    def estimate_success_probability(self, criteria: Dict, sample_size: int = 2000) -> Dict:
        size = criteria.get('size', 6)
        passed = 0
        target_avg = criteria.get('target_average', self.analyzer.global_avg_sum)
        sum_tolerance = LotteryConfig.DEFAULT_SUM_TOLERANCE

        for _ in range(sample_size):
            candidate = sorted(random.sample(self.full_pool, size))
            inc_draw = criteria.get('include_from_draw')
            inc_cnt = criteria.get('include_count', 0)
            if inc_draw and inc_cnt > 0:
                draw_nums = self.analyzer.get_numbers_from_draw(inc_draw)
                if draw_nums:
                    intersect = len(set(candidate) & set(draw_nums))
                    if intersect != inc_cnt: continue 

            if criteria.get('sequences_count') is not None and TicketValidator.count_sequences(candidate) != criteria['sequences_count']: continue
            if criteria.get('odd_count') is not None and sum(1 for n in candidate if n%2!=0) != criteria['odd_count']: continue
            if criteria.get('shadows_count') is not None:
                curr = TicketValidator.count_shadow_occurrences(candidate)
                if not (max(0, criteria['shadows_count']-1) <= curr <= criteria['shadows_count']+1): continue
            if criteria.get('sum_near_avg'):
                s = sum(candidate)
                if not (target_avg * (1-sum_tolerance) <= s <= target_avg * (1+sum_tolerance)): continue
            passed += 1
        
        prob = (passed / sample_size) * 100
        advice = "✅ سهلة" if prob > 5 else ("⚡ متوسطة" if prob > 0.5 else "⚠️ صعبة جداً")
        return {"probability": round(prob, 2), "advice": advice}

    def _generate_single_ticket(self, criteria: Dict, sum_tolerance: float) -> Dict:
        try:
            size = criteria.get('size', 6)
            strategy = criteria.get('strategy', 'Any')
            target_avg = criteria.get('target_average', self.analyzer.global_avg_sum)
            
            include_draw_id = criteria.get('include_from_draw')
            include_count = criteria.get('include_count', 0)
            forced_numbers = []
            forbidden_numbers = set()

            if include_draw_id and include_count > 0:
                source_nums = self.analyzer.get_numbers_from_draw(include_draw_id)
                if source_nums:
                    forced_numbers = random.sample(source_nums, min(len(source_nums), include_count))
                    forbidden_numbers = set(source_nums) - set(forced_numbers)
            
            pool_source = self.full_pool
            if strategy == 'Hot': pool_source = list(self.analyzer.hot_pool)
            elif strategy == 'Cold': pool_source = list(self.analyzer.cold_pool)
            
            current_pool = [n for n in pool_source if n not in forced_numbers and n not in forbidden_numbers]
            needed_count = size - len(forced_numbers)
            if len(current_pool) < needed_count:
                current_pool = [n for n in self.full_pool if n not in forced_numbers and n not in forbidden_numbers]

            for attempt in range(LotteryConfig.MAX_GENERATION_ATTEMPTS):
                random_part = random.sample(current_pool, needed_count)
                candidate = sorted(forced_numbers + random_part)
                
                if strategy == 'Balanced':
                    hot_in_ticket = sum(1 for n in candidate if n in self.analyzer.hot_pool)
                    half = size / 2
                    if not (half - 1 <= hot_in_ticket <= half + 1): continue

                if criteria.get('sequences_count') is not None and TicketValidator.count_sequences(candidate) != criteria['sequences_count']: continue
                if criteria.get('odd_count') is not None and sum(1 for n in candidate if n%2!=0) != criteria['odd_count']: continue
                if criteria.get('shadows_count') is not None:
                    curr = TicketValidator.count_shadow_occurrences(candidate)
                    if attempt < LotteryConfig.STRICT_SHADOW_ATTEMPTS:
                        if curr != criteria['shadows_count']: continue
                    else:
                        if not (max(0, criteria['shadows_count']-1) <= curr <= criteria['shadows_count']+1): continue
                if criteria.get('sum_near_avg'):
                    s = sum(candidate)
                    if not (target_avg * (1-sum_tolerance) <= s <= target_avg * (1+sum_tolerance)): continue
                if criteria.get('anti_match_limit') and not TicketValidator.check_anti_match_optimized(set(candidate), self.analyzer, criteria['anti_match_limit']): continue
                
                return {"status": "success", "ticket": candidate, "attempts": attempt + 1}
            return {"status": "error", "reason": "استنفاد المحاولات"}
        except Exception as e:
            return {"status": "error", "reason": f"خطأ داخلي: {str(e)}"}

    def generate_batch(self, criteria: Dict, count: int = 1) -> Dict:
        errors = self._validate_criteria(criteria)
        if errors: return {"status": "validation_error", "errors": errors}
        
        actual_count = min(count, LotteryConfig.MAX_BATCH_SIZE)
        generated_tickets = []
        seen_signatures = set()
        errors_list = []
        
        progress_bar = st.progress(0)
        
        for i in range(actual_count):
            for retry in range(50):
                res = self._generate_single_ticket(criteria, LotteryConfig.DEFAULT_SUM_TOLERANCE)
                if res['status'] == 'success':
                    sig = tuple(res['ticket'])
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        anl = TicketValidator.analyze_ticket(res['ticket'], self.analyzer)
                        generated_tickets.append({"id": i+1, "numbers": res['ticket'], "analysis": anl, "attempts": res['attempts']})
                        break
                else:
                    if retry == 0: errors_list.append(res['reason'])
            
            progress_bar.progress((i + 1) / actual_count)
        
        progress_bar.empty()
        status = "success" if len(generated_tickets) == actual_count else ("partial_success" if generated_tickets else "failed")
        return {"status": status, "requested": count, "generated": len(generated_tickets), "tickets": generated_tickets, "errors": Counter(errors_list).most_common(3)}

# ==============================================================================
# 5. واجهة المستخدم (v9.0 Strict Mode)
# ==============================================================================
def main():
    st.set_page_config(page_title="نظام لوتري الأردن الذكي", page_icon="🎰", layout="wide", initial_sidebar_state="expanded")
    
    initialize_session_state()

    custom_css = """
    <style>
    .main { direction: rtl; }
    h1, h2, h3, p, div, label, span { text-align: right; font-family: 'Segoe UI', sans-serif; }
    .stMetric { text-align: right !important; }
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #f0f2f6; color: #333;
        text-align: center; padding: 10px;
        border-top: 1px solid #ddd; font-size: 14px;
        z-index: 999; font-family: 'Segoe UI', sans-serif; font-weight: bold;
    }
    .file-warning {
        padding: 20px;
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        text-align: center;
        margin-top: 20px;
        font-size: 18px;
    }
    @media (prefers-color-scheme: dark) {
        .footer { background-color: #0e1117; color: #888; border-top: 1px solid #333; }
        .file-warning { background-color: #721c24; color: #f8d7da; border: 1px solid #f5c6cb; }
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    st.title("🎯 القناص لتوليد وفحص تذاكر لوتري الأردن")
    
    # التحميل التلقائي من GitHub عند أول دخول
    if not st.session_state.auto_loaded and st.session_state.history_df is None:
        with st.spinner("🔄 جاري تحميل البيانات من GitHub..."):
            df, msg = load_from_github()
            if df is not None:
                st.session_state.history_df = df
                st.session_state.analyzer = LotteryAnalyzer(df)
                st.session_state.generator = TicketGenerator(st.session_state.analyzer)
            st.session_state.auto_loaded = True
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("1. إعدادات البيانات")
        
        # إذا تم تحميل البيانات تلقائياً، أظهر معلومات
        if st.session_state.history_df is not None and st.session_state.auto_loaded:
            st.success("✅ البيانات محملة من GitHub")
            if st.button("🔄 إعادة تحميل من GitHub"):
                with st.spinner("جاري إعادة التحميل..."):
                    df, msg = load_from_github()
                    if df is not None:
                        st.session_state.history_df = df
                        st.session_state.analyzer = LotteryAnalyzer(df)
                        st.session_state.generator = TicketGenerator(st.session_state.analyzer)
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
        
        # خيار تحميل ملف بديل
        st.markdown("---")
        st.caption("أو ارفع ملف بديل:")
        uploaded_file = st.file_uploader("📂 رفع ملف Excel/CSV", type=['xlsx', 'csv'])
        
        df = None
        msg = ""
        
        if uploaded_file:
            df, msg = load_and_process_data(uploaded_file)
            if df is not None:
                st.success("✅ تم تحميل الملف المرفوع")
            else:
                st.error(msg)
        elif st.session_state.history_df is None:
            # محاولة البحث عن ملف محلي فقط
            for fname in ['data.xlsx', 'data.csv', 'lotto.xlsx', 'lotto.csv']:
                if os.path.exists(fname):
                    df, msg = load_and_process_data(fname)
                    if df is not None: st.info("📂 تم تحميل الملف التلقائي")
                    break

        if df is not None:
            st.session_state.history_df = df
            # تهيئة المحلل فقط عند وجود بيانات حقيقية
            if st.session_state.analyzer is None or len(st.session_state.analyzer.history_df) != len(df):
                st.session_state.analyzer = LotteryAnalyzer(df)
                st.session_state.generator = TicketGenerator(st.session_state.analyzer)
            
            analyzer = st.session_state.analyzer
            st.metric("إجمالي السحوبات", analyzer.total_draws)
            st.metric("المتوسط العام", f"{analyzer.global_avg_sum:.2f}")
        elif st.session_state.history_df is not None:
            analyzer = st.session_state.analyzer
            st.metric("إجمالي السحوبات", analyzer.total_draws)
            st.metric("المتوسط العام", f"{analyzer.global_avg_sum:.2f}")
        
        # معلومات عن التحديث
        st.markdown("---")
        st.info("""
        📅 **جدول السحوبات:**
        - الأحد من كل أسبوع
        - الأربعاء من كل أسبوع
        
        💡 اضغط "🔄 إعادة تحميل" بعد كل سحب جديد
        """)

    # --- MAIN CONTENT CONTROL ---
    # إذا لم توجد بيانات حقيقية، نوقف التطبيق هنا
    if st.session_state.history_df is None:
        st.markdown('<div class="file-warning">⛔ فشل تحميل البيانات من GitHub.<br>الرجاء تحميل ملف بيانات السحوبات (Excel) من القائمة الجانبية للبدء.</div>', unsafe_allow_html=True)
        st.stop() # إيقاف التنفيذ

    # إذا وصلنا هنا، فالبيانات حقيقية 100%
    analyzer = st.session_state.analyzer
    generator = st.session_state.generator

    # --- TABS ---
    tab1, tab2 = st.tabs(["🚀 توليد تذاكر جديدة", "🔍 فحص تذكرة تاريخي"])

    # --------------------------------------------------------
    # Tab 1: Generator
    # --------------------------------------------------------
    with tab1:
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.subheader("⚙️ إعدادات التوليد")
            strategy = st.selectbox("🎯 استراتيجية الأرقام", ["Any (الكل)", "Hot (ساخنة)", "Cold (باردة)", "Balanced (متوازنة)"])
            strategy_map = {"Any (الكل)": "Any", "Hot (ساخنة)": "Hot", "Cold (باردة)": "Cold", "Balanced (متوازنة)": "Balanced"}
            
            with st.container(border=True):
                st.markdown("**📊 ضبط المتوسط الحسابي**")
                avg_chk = st.checkbox("الالتزام بالمتوسط", value=True)
                target_avg_val = analyzer.global_avg_sum
                chart_data = [] 
                if avg_chk:
                    avg_mode = st.selectbox("المرجع لحساب المتوسط:", ["كافة السحوبات (Default)", "آخر N سحب", "نطاق محدد"])
                    if avg_mode == "آخر N سحب":
                        n_draws = st.number_input("عدد السحوبات الأخيرة", 5, analyzer.total_draws, 20)
                        target_avg_val, chart_data = analyzer.calculate_custom_average("Last N Draws", param1=n_draws)
                        st.caption(f"متوسط آخر {n_draws} سحب: **{target_avg_val:.2f}**")
                    elif avg_mode == "نطاق محدد":
                        c1, c2 = st.columns(2)
                        start_d = c1.number_input("من سحب", 1, analyzer.total_draws, max(1, analyzer.total_draws-50))
                        end_d = c2.number_input("إلى سحب", 1, analyzer.total_draws, analyzer.total_draws)
                        target_avg_val, chart_data = analyzer.calculate_custom_average("Specific Range", param1=start_d, param2=end_d)
                        st.caption(f"المتوسط للنطاق: **{target_avg_val:.2f}**")
                    else:
                        target_avg_val, chart_data = analyzer.calculate_custom_average("All")
                        st.caption(f"المتوسط العام: **{target_avg_val:.2f}**")
                    if chart_data: st.line_chart(chart_data, height=150)

            with st.container(border=True):
                t_count = st.number_input("عدد التذاكر المراد توليدها", 1, 10, 3)
                t_size = st.slider("حجم التذكرة المراد توليدها", 6, 10, 6)
                odd = st.number_input("عدد الفردي", 0, t_size, t_size//2)
                seq = st.number_input("عدد المتتاليات في كل تذكرة مراد توليدها", 0, t_size-1, 0)
                sha = st.number_input("عدد الظلال في كل تذكرة مراد توليدها", 0, 3, 1)

            with st.container(border=True):
                st.markdown("**🔄 تكرار صارم (Pivot)**")
                use_past = st.checkbox("تثبيت أرقام من سحب سابق")
                inc_draw = None; inc_cnt = 0
                if use_past:
                    c1, c2 = st.columns(2)
                    inc_draw = c1.number_input("رقم السحب", 1, analyzer.total_draws, analyzer.total_draws)
                    inc_cnt = c2.number_input("عدد الأرقام", 1, min(6, t_size), 1)
                    past_nums = analyzer.get_numbers_from_draw(inc_draw)
                    if past_nums: st.caption(f"أرقام السحب {inc_draw}: {past_nums}")

            st.markdown("---")
            anti = st.slider("تجنب تطابق (عدد أرقام) مع أي نتيجة سحب سابق", 3, t_size, 5)

            criteria = {
                'size': t_size, 'sequences_count': seq, 'odd_count': odd, 
                'shadows_count': sha, 'anti_match_limit': anti, 'sum_near_avg': avg_chk,
                'target_average': target_avg_val,
                'include_from_draw': inc_draw if use_past else None, 'include_count': inc_cnt if use_past else 0,
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
                    st.error("خطأ:"); [st.write(f"- {e}") for e in res['errors']]
                elif res['status'] == 'failed':
                    st.error("فشل التوليد."); st.write("الأسباب:", res['errors'])
                else:
                    if res['status'] == 'partial_success': st.warning(f"تم توليد {res['generated']} تذاكر فقط.")
                    else: st.success(f"تم توليد {res['generated']} تذاكر بنجاح!")
                    
                    for t in res['tickets']:
                        with st.expander(f"🎫 تذكرة #{t['id']} - {t['analysis']['profile']}", expanded=True):
                            st.markdown("".join([f"<span style='display:inline-block; background:#dcfce7; color:#166534; padding:5px 10px; margin:2px; border-radius:50%; font-weight:bold; border:1px solid #166534'>{n}</span>" for n in t['numbers']]), unsafe_allow_html=True)
                            ca, cb, cc = st.columns(3)
                            ca.caption(f"المجموع: {t['analysis']['sum']}")
                            cb.caption(f"المتتاليات: {t['analysis']['sequences']}")
                            cc.caption(f"الظلال: {t['analysis']['shadows']}")
                            if use_past and inc_draw:
                                draw_nums = set(analyzer.get_numbers_from_draw(inc_draw))
                                matches = set(t['numbers']) & draw_nums
                                color = "green" if len(matches)==inc_cnt else "red"
                                st.markdown(f":{color}[✅ المطلوب: {inc_cnt} | 🎯 المحقق: {len(matches)} ({list(matches)})]")

    # --------------------------------------------------------
    # Tab 2: Checker
    # --------------------------------------------------------
    with tab2:
        st.subheader("🕵️ فحص تذكرة تاريخياً")
        c_check1, c_check2 = st.columns([1, 2])
        with c_check1:
            chk_size = st.radio("حدد حجم التذكرة للفحص:", [6, 7, 8, 9, 10], horizontal=True)
        
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
                        with st.expander(f"🌟 تطابق {count} أرقام (عدد المرات: {len(res_list)})", expanded=True):
                            for item in res_list:
                                st.markdown(f"- **سحب رقم {item['draw_id']}:** الأرقام المتطابقة {item['matched_nums']}")
                if not found_any: st.info("✅ هذه التذكرة نظيفة! (لم تحقق 4,5,6 سابقاً)")

                st.divider()

                # 2. Frequency
                st.markdown("### 2️⃣ تحليل تكرار الأرقام")
                freq_df = analyzer.get_numbers_frequency_stats(sorted_chk)
                col_f1, col_f2 = st.columns([1, 2])
                with col_f1: st.dataframe(freq_df, hide_index=True, use_container_width=True)
                with col_f2: st.bar_chart(freq_df.set_index('الرقم')['عدد مرات الظهور'], color="#166534")

                st.divider()

                # 3. Sequences
                st.markdown("### 3️⃣ فحص المتتاليات")
                seq_results = analyzer.analyze_sequences_history(sorted_chk)
                if not seq_results: st.write("🔹 لا توجد متتاليات.")
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

    st.markdown("""<div class="footer">برمجة وتطوير: <b>محمد العمري</b></div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
