"""
Task management for expense document processing
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TaskManager:
    """Manage extraction tasks for expense documents"""
    
    def __init__(self):
        """Initialize task manager with predefined tasks"""
        
        self.tasks = {
            "expense_amount": {
                "name": "expense_amount",
                "description": "Extract the total expense amount from the document",
                "prompt_template": """
                Based on the following document content, extract the total expense amount.
                Look for monetary values, totals, amounts due, or similar financial figures.
                
                Document content:
                {context}
                
                Please provide:
                1. The total expense amount (number only, no currency symbol)
                2. The currency if identifiable
                3. Your confidence level (high/medium/low)
                
                Format your response as JSON:
                {{
                    "amount": "number_value",
                    "currency": "currency_code_or_symbol",
                    "confidence": "high/medium/low",
                    "source_text": "exact_text_where_found"
                }}
                """,
                "output_format": "json"
            },
            
            "vendor_name": {
                "name": "vendor_name",
                "description": "Extract the vendor/company name from the document",
                "prompt_template": """
                Based on the following document content, extract the vendor or company name.
                Look for business names, company names, merchant names, or service providers.
                
                Document content:
                {context}
                
                Please provide:
                1. The vendor/company name
                2. Your confidence level (high/medium/low)
                
                Format your response as JSON:
                {{
                    "vendor_name": "extracted_name",
                    "confidence": "high/medium/low",
                    "source_text": "exact_text_where_found"
                }}
                """,
                "output_format": "json"
            },
            
            "expense_date": {
                "name": "expense_date",
                "description": "Extract the expense date from the document",
                "prompt_template": """
                Based on the following document content, extract the expense or transaction date.
                Look for dates, timestamps, or date ranges indicating when the expense occurred.
                
                Document content:
                {context}
                
                Please provide:
                1. The expense date in YYYY-MM-DD format
                2. Your confidence level (high/medium/low)
                
                Format your response as JSON:
                {{
                    "expense_date": "YYYY-MM-DD",
                    "confidence": "high/medium/low",
                    "source_text": "exact_text_where_found"
                }}
                """,
                "output_format": "json"
            },
            
            "expense_category": {
                "name": "expense_category",
                "description": "Categorize the expense type",
                "prompt_template": """
                Based on the following document content, categorize the type of expense.
                Common categories include: Travel, Meals, Office Supplies, Software, Professional Services, etc.
                
                Document content:
                {context}
                
                Please provide:
                1. The expense category
                2. Your confidence level (high/medium/low)
                3. A brief justification
                
                Format your response as JSON:
                {{
                    "category": "category_name",
                    "confidence": "high/medium/low",
                    "justification": "brief_explanation"
                }}
                """,
                "output_format": "json"
            },
            
            "line_items": {
                "name": "line_items",
                "description": "Extract individual line items from the document",
                "prompt_template": """
                Based on the following document content, extract individual line items if present.
                Look for itemized lists, product names, services, quantities, and individual prices.
                
                Document content:
                {context}
                
                Please provide a list of line items found:
                
                Format your response as JSON:
                {{
                    "line_items": [
                        {{
                            "description": "item_description",
                            "quantity": "quantity_if_available",
                            "unit_price": "price_per_unit",
                            "total_price": "total_for_this_item"
                        }}
                    ],
                    "confidence": "high/medium/low"
                }}
                """,
                "output_format": "json"
            },
            
            "tax_amount": {
                "name": "tax_amount",
                "description": "Extract tax amount from the document",
                "prompt_template": """
                Based on the following document content, extract tax information.
                Look for tax amounts, tax rates, VAT, sales tax, or similar tax-related information.
                
                Document content:
                {context}
                
                Please provide:
                1. The tax amount
                2. The tax rate if available
                3. The tax type (sales tax, VAT, etc.)
                
                Format your response as JSON:
                {{
                    "tax_amount": "amount_value",
                    "tax_rate": "rate_percentage",
                    "tax_type": "type_of_tax",
                    "confidence": "high/medium/low",
                    "source_text": "exact_text_where_found"
                }}
                """,
                "output_format": "json"
            },
            
            "document_summary": {
                "name": "document_summary",
                "description": "Generate a summary of the expense document",
                "prompt_template": """
                Based on the following document content, provide a comprehensive summary.
                Include key information about the expense, vendor, amount, and purpose.
                
                Document content:
                {context}
                
                Please provide a concise summary covering:
                1. What the expense is for
                2. Who the vendor is
                3. The amount and date
                4. Any other relevant details
                
                Format your response as JSON:
                {{
                    "summary": "comprehensive_summary_text",
                    "key_points": ["point1", "point2", "point3"],
                    "confidence": "high/medium/low"
                }}
                """,
                "output_format": "json"
            }
        }
        
        logger.info(f"TaskManager initialized with {len(self.tasks)} tasks")
    
    def get_task(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get task definition by name"""
        return self.tasks.get(task_name)
    
    def get_task_info(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get task information by name (alias for get_task for compatibility)"""
        task = self.get_task(task_name)
        if task:
            return {
                "query": task["description"],  # Map description to query field
                "description": task["description"],
                "prompt_template": task["prompt_template"],
                "output_format": task["output_format"]
            }
        return None
    
    def list_tasks(self) -> List[str]:
        """Get list of available task names"""
        return list(self.tasks.keys())
    
    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all task definitions"""
        return self.tasks.copy()
    
    def add_custom_task(self, task_name: str, description: str, 
                       prompt_template: str, output_format: str = "json") -> bool:
        """Add a custom task definition"""
        
        try:
            if task_name in self.tasks:
                logger.warning(f"Task {task_name} already exists, overwriting")
            
            self.tasks[task_name] = {
                "name": task_name,
                "description": description,
                "prompt_template": prompt_template,
                "output_format": output_format
            }
            
            logger.info(f"Added custom task: {task_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding custom task {task_name}: {e}")
            return False
    
    def remove_task(self, task_name: str) -> bool:
        """Remove a task definition"""
        
        try:
            if task_name in self.tasks:
                del self.tasks[task_name]
                logger.info(f"Removed task: {task_name}")
                return True
            else:
                logger.warning(f"Task {task_name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error removing task {task_name}: {e}")
            return False
    
    def validate_task(self, task_name: str) -> bool:
        """Validate if a task exists and is properly configured"""
        
        if task_name not in self.tasks:
            return False
        
        task = self.tasks[task_name]
        required_fields = ["name", "description", "prompt_template", "output_format"]
        
        for field in required_fields:
            if field not in task or not task[field]:
                logger.error(f"Task {task_name} missing required field: {field}")
                return False
        
        return True
    
    def format_prompt(self, task_name: str, context: str) -> Optional[str]:
        """Format the prompt template with context"""
        
        if not self.validate_task(task_name):
            return None
        
        try:
            task = self.tasks[task_name]
            prompt = task["prompt_template"].format(context=context)
            return prompt
            
        except Exception as e:
            logger.error(f"Error formatting prompt for task {task_name}: {e}")
            return None
