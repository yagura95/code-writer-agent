from typing import TypedDict, Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import subprocess
import tempfile
import os
import asyncio

class State(TypedDict):
    task_description: str
    current_code: Optional[str]
    error_message: Optional[str]
    attempt_count: int
    success: bool
    execution_output: Optional[str]

async def write_code(state: State) -> State:
    """Generate initial code based on task description."""
    llm = Ollama(model="qwen2.5-coder:3b")
    
    prompt = PromptTemplate.from_template("""You are an expert Python programmer. Write code that accomplishes the given task.
    Include a test case at the end that prints 'TEST PASSED' if successful.
    Only return the Python code, nothing more. 
    
    Task: {task}
    """)
    
    # Wait for LLM response
    print("Generating initial code...")
    response = await llm.ainvoke(prompt.format(task=state["task_description"]))
    response = response.strip().split("\n")[1:-1]

    state["current_code"] = "\n".join(response)
    state["attempt_count"] += 1
    print(f"\nAttempt {state['attempt_count']} - Generated code:")
    print(state["current_code"])
    return state

async def test_code(state: State) -> State:
    """Test the generated code by running it."""
    if not state["current_code"]:
        state["error_message"] = "No code to test"
        return state
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(state["current_code"])
        temp_file = f.name

    """For debuggin purposes"""
    f = open(str(state["attempt_count"]) + ".txt", "w")
    f.write(state["current_code"])
    f.close()
    
    try:
        print("\nTesting code...")
        process = await asyncio.create_subprocess_exec(
            'python3', temp_file,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5.0)
        except asyncio.TimeoutError:
            process.kill()
            state["error_message"] = "Code execution timed out"
            state["success"] = False
            print("Execution timed out")
            return state
            
        state["execution_output"] = stdout.decode()
        print(f"Output: {state['execution_output']}")
        
        if process.returncode == 0 and "TEST PASSED" in state["execution_output"]:
            state["success"] = True
            state["error_message"] = None
            print("Test successful!")
        else:
            state["error_message"] = stderr.decode() or "Test failed"
            state["success"] = False
            print(f"Test failed: {state['error_message']}")
            
    finally:
        os.unlink(temp_file)
    
    return state

async def fix_code(state: State) -> State:
    """Fix the code based on error message."""
    llm = Ollama(model="qwen2.5-coder:3b")
    
    prompt = PromptTemplate.from_template("""Fix this Python code based on the error message.
    Include a test case at the end that prints 'TEST PASSED' if successful.
    Only return the complete fixed, nothing more.
    
    Task: {task}
    Current code:
    {code}
    
    Error message:
    {error}
    """)
    
    print("\nGenerating fix...")
    response = await llm.ainvoke(
        prompt.format(
            task=state["task_description"],
            code=state["current_code"],
            error=state["error_message"]
        )
    )
    response = response.strip().split("\n")
    
    state["current_code"] = "\n".join(response) 
    state["attempt_count"] += 1
    print(f"\nAttempt {state['attempt_count']} - Fixed code:")
    print(state["current_code"])
    return state

async def run_workflow(state: State) -> State:
    """Run the workflow without using LangGraph."""
    # Initial code generation
    state = await write_code(state)
    
    max_attempts = 3
    while state["attempt_count"] < max_attempts:
        # Test the code
        state = await test_code(state)
        
        # If successful, we're done
        if state["success"]:
            break
            
        # If not successful and we haven't hit max attempts, try to fix
        if state["attempt_count"] < max_attempts:
            state = await fix_code(state)
    
    return state

async def write_and_test_code(task_description: str) -> State:
    """Main function to write and test code for a given task."""
    # Create initial state
    initial_state = {
        "task_description": task_description,
        "current_code": None,
        "error_message": None,
        "attempt_count": 0,
        "success": False,
        "execution_output": None
    }
    
    # Run the workflow
    final_state = await run_workflow(initial_state)
    
    # Print final results
    if final_state["success"]:
        print("\n✅ Final working code:")
        print(final_state["current_code"])
        print("\nExecution output:")
        print(final_state["execution_output"])
    else:
        print("\n❌ Failed to generate working code after", final_state["attempt_count"], "attempts")
        print("\nLast error:")
        print(final_state["error_message"])
        print("\nLast code attempt:")
        print(final_state["current_code"])
    
    return final_state

# Example usage
if __name__ == "__main__":
    task = """Write a Python function called sum_even_numbers that:
    1. Takes a list of numbers as input
    2. Returns the sum of all even numbers in the list
    Include a test case that tests the function with [1,2,3,4] and prints 'TEST PASSED' if the result is 6"""
    
    asyncio.run(write_and_test_code(task))
