def check(func, *args, **kw):
    try:
        func(*args, **kw)
        return True
    except Exception:
        return False