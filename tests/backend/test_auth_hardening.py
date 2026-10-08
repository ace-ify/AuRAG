"""Auth hardening (step 5): production config guard, no test-JWT backdoor in
production, and claims-only tenant context when auth is enabled."""
import pytest

from backend.app.core import auth


def test_assert_secure_config_blocks_prod_without_auth(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    with pytest.raises(RuntimeError, match="AUTH_ENABLED"):
        auth.assert_secure_config()


def test_assert_secure_config_allows_prod_with_auth(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUTH_ENABLED", "true")
    auth.assert_secure_config()  # does not raise


def test_assert_secure_config_noop_outside_prod(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    auth.assert_secure_config()  # dev/local: fine


def test_test_jwt_secret_ignored_in_production(monkeypatch):
    import jwt
    secret = "aurag-test-secret-32-chars-long!!"
    monkeypatch.setenv("TEST_JWT_SECRET", secret)
    monkeypatch.setenv("APP_ENV", "production")
    token = jwt.encode({"sub": "forged"}, secret, algorithm="HS256")

    # In production the HS256 dev path is skipped; with no JWKS match this must
    # fail rather than accept the forged token.
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        auth.verify_entra_token(token)


def test_site_id_not_taken_from_header_when_auth_enabled(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")

    class _Req:
        headers = {"X-Site-ID": "plant-attacker-99", "X-Organization-ID": "evil-corp"}

    user = auth.claims_to_user_profile({"sub": "u1"}, request=_Req())
    assert user.site_id == "plant-mumbai-01"      # default, NOT the spoofed header
    assert user.organization_id == "aurag-industrial"


def test_site_id_from_header_allowed_in_dev(monkeypatch):
    monkeypatch.delenv("AUTH_ENABLED", raising=False)  # dev/local

    class _Req:
        headers = {"X-Site-ID": "plant-jamnagar-02"}

    user = auth.claims_to_user_profile({"sub": "u1"}, request=_Req())
    assert user.site_id == "plant-jamnagar-02"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
