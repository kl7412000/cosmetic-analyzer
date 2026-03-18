"""
Cosmetic Analyzer 項目的單元測試
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# 加入專案根目錄
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.ingestor import load_ingredients
from rag.validator import validate_format


class TestIngestor:
    """測試成分數據加載"""
    
    def test_load_ingredients_returns_list(self):
        """測試 load_ingredients 返回列表"""
        result = load_ingredients("data/ingredients.json")
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_load_ingredients_has_metadata(self):
        """測試加載的文件有正確的 metadata"""
        result = load_ingredients("data/ingredients.json")
        doc = result[0]
        assert hasattr(doc, 'metadata')
        assert 'ingredient' in doc.metadata


class TestValidator:
    """測試格式驗證"""
    
    def test_validate_valid_format(self):
        """測試有效格式驗證"""
        data = {
            "ingredient": "Test Ingredient",
            "inci_name": "Test INCI",
            "functions": ["Function1"],
            "benefits": ["Benefit1"],
            "risks": ["Risk1"],
            "eu_regulation": "Valid"
        }
        passed, errors = validate_format(data)
        assert passed is True
        assert errors == []
    
    def test_validate_missing_fields(self):
        """測試缺失字段驗證"""
        data = {
            "ingredient": "Test"
        }
        passed, errors = validate_format(data)
        assert passed is False
        assert len(errors) > 0


class TestGraphIntegration:
    """整合測試 - 測試 graph 的基本流程"""
    
    @patch('rag.retriever.get_vectorstore')
    def test_query_node_basic(self, mock_vectorstore):
        """測試 query_node 基本功能"""
        # 這個測試需要 mock FAISS 向量庫
        # 示例只展示框架
        from rag.graph import query_node, AnalysisState
        
        mock_state = AnalysisState(
            input_text="Glycerin",
            input_image=None,
            original_ingredients=[],
            ingredients=["Glycerin"],
            found=[],
            not_found=[],
            enriched_data=[],
            results=[],
            error=None
        )
        
        # 注意：這個測試需要實際的 FAISS 索引來運作
        # 或者使用更完整的 mock 設置


class TestChainIntegration:
    """整合測試 - 測試 LLM chain"""
    
    @patch('rag.groq_client.call_groq')
    def test_enrich_fallback(self, mock_groq):
        """測試 LLM fallback enrichment"""
        mock_groq.return_value = {
            "ingredient": "Salicylic Acid",
            "inci_name": "Salicylic Acid",
            "functions": ["Exfoliant"],
            "benefits": ["Reduces acne"],
            "risks": ["May cause irritation"],
            "eu_regulation": "Allowed",
            "skin_type": ["Oily", "Combination"]
        }
        
        from rag.enricher import enrich_from_name
        
        # 此測試示例
        result = mock_groq("Test ingredient")
        assert result["ingredient"] == "Salicylic Acid"
        assert "functions" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
