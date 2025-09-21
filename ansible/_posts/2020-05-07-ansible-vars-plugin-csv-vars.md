csv 文件示例

# vars plugins: csv_vars

## Requirement

Parse data from CSV into host variables.

CSV file example:

```csv
hostname,gateway,vlan30,vlan40,vlan50,vlan60,vlan70
localhost,192.168.30.254,192.168.30.31,192.168.40.31,192.168.50.31,192.168.60.31,192.168.70.31
```

- Place the CSV file in the `csv_vars` directory under `inventory` or `playbook`, and it will be automatically parsed.
- The `hostname` field in the CSV must match the host name in the inventory.
- It is recommended to name the CSV as GROUP_OR_HOST_NAME.csv. Multiple CSV files are supported, and variable overriding is also supported.

## Write the Plugin

File path: `vars_plugins/csv_vars.py`

```python
...existing code...
```
> Code from https://github.com/guozijn/csv_vars



## CSV File

Place the CSV file in the `csv_vars` directory under inventory or playbook.

```bash
mdkir csv_vars
cat << EOF >> csv_vars/nodes.csv
hostname,gateway,vlan30,vlan40,vlan50,vlan60,vlan70
192.168.77.130,192.168.30.254,192.168.30.31,192.168.40.31,192.168.50.31,192.168.60.31,192.168.70.31
EOF
```

## Run Playbook

```yaml
# cat test_vars.yml
{% raw %}
- hosts: 192.168.77.130
  gather_facts: no
  tasks:
    - debug:
        msg: "{{ lookup('vars', item) }}"
      loop: "{{ hostvars[inventory_hostname].keys() | select('match', '^vlan.*$|gateway') | list }}"
{% endraw %}
```

Execution result:

```bash
# ansible-playbook test_vars.yml

PLAY [192.168.77.130] ************************************************************************************************

TASK [debug] *********************************************************************************************************
ok: [192.168.77.130] => (item=gateway) => {
    "msg": "192.168.30.254"
}
ok: [192.168.77.130] => (item=vlan60) => {
    "msg": "192.168.60.31"
}
ok: [192.168.77.130] => (item=vlan30) => {
    "msg": "192.168.30.31"
}
ok: [192.168.77.130] => (item=vlan40) => {
    "msg": "192.168.40.31"
}
ok: [192.168.77.130] => (item=vlan70) => {
    "msg": "192.168.70.31"
}
ok: [192.168.77.130] => (item=vlan50) => {
    "msg": "192.168.50.31"
}

PLAY RECAP ***********************************************************************************************************
192.168.77.130             : ok=1    changed=0    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   

```

## Run ad-hoc

Configure `ansible.cfg` to set the custom plugin directory:

```ini
[defaults]
vars_plugins       = /etc/ansible/vars_plugins
```

```bash
# ansible 192.168.77.130 -m debug -a 'var=gateway'
192.168.77.130 | SUCCESS => {
    "gateway": "192.168.30.254"
}

# ansible 192.168.77.130 -m debug -a 'var=vlan30'
192.168.77.130 | SUCCESS => {
    "vlan30": "192.168.30.31"
}
```
